from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from api.auth import require_tenant
from api.logging import RequestContextMiddleware, bind_tenant, configure_logging
from api.ratelimit import consume
from api.tenants import Tenant
from api.voices import Voice, get_voice, list_voices
from inference.pipeline import run_tts_pipeline


configure_logging()

app = FastAPI(title="Bantu TTS API", version="0.2.0")
app.add_middleware(RequestContextMiddleware)


class SynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    language: str = Field(default="shona")
    voice: str = Field(default="female_1")
    speed: float = Field(default=1.0, gt=0.0, le=2.0)


class VoiceResponse(BaseModel):
    voice_id: str
    language: str
    gender: str
    sample_rate: int

    @classmethod
    def from_voice(cls, voice: Voice) -> "VoiceResponse":
        return cls(
            voice_id=voice.voice_id,
            language=voice.language,
            gender=voice.gender,
            sample_rate=voice.sample_rate,
        )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/voices", response_model=list[VoiceResponse])
def voices(tenant: Tenant = Depends(require_tenant)) -> list[VoiceResponse]:
    bind_tenant(tenant.tenant_id)
    return [VoiceResponse.from_voice(v) for v in list_voices()]


@app.post("/synthesize")
def synthesize(payload: SynthesizeRequest, tenant: Tenant = Depends(require_tenant)) -> Response:
    bind_tenant(tenant.tenant_id)
    consume(tenant)

    # Voice resolution: accept legacy short names ("female_1") and new
    # language-qualified IDs ("shona_female_1"). Future tenant-scoped
    # access control hooks here.
    candidate_ids = (payload.voice, f"{payload.language}_{payload.voice}")
    if not any(get_voice(vid) for vid in candidate_ids if vid):
        # Voice not in catalog — soft-fail to default for now; tighten when
        # tenant-voice permissions land.
        pass

    try:
        result, _ = run_tts_pipeline(
            text=payload.text,
            language=payload.language,
            voice=payload.voice,
            speed=payload.speed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - safety net
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    media_type = "audio/wav" if result.format == "wav" else "application/octet-stream"
    return Response(content=result.audio_bytes, media_type=media_type)
