from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from inference.pipeline import run_tts_pipeline


app = FastAPI(title="Bantu TTS API", version="0.1.0")


class SynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    language: str = Field(default="shona")
    voice: str = Field(default="female_1")
    speed: float = Field(default=1.0, gt=0.0, le=2.0)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/synthesize")
def synthesize(payload: SynthesizeRequest) -> Response:
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
