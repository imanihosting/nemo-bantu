"""Bantu TTS API — FastAPI backend serving Shona text-to-speech.

Run with:
    cd /home/blaquesoul/Desktop/nemo-bantu
    .venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

from inference.pipeline import run_tts_pipeline
from inference.synthesize import NemoSynthesizer

app = FastAPI(title="Bantu TTS API", version="0.2.0")

# Allow browser access from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    language: str = Field(default="shona")
    voice: str = Field(default="female_1")
    speed: float = Field(default=1.0, gt=0.0, le=2.0)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/reload")
def reload_models() -> dict:
    """Reload models from the latest training checkpoints."""
    synth = NemoSynthesizer.reload()
    return {"status": "reloaded", "models_loaded": synth.is_loaded}


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
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    media_type = "audio/wav" if result.format == "wav" else "application/octet-stream"
    return Response(content=result.audio_bytes, media_type=media_type)


# ── Built-in Web UI ──────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def web_ui():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bantu TTS — Shona Text-to-Speech</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'Inter', sans-serif;
    min-height: 100vh;
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #e0e0e0;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 2rem;
  }

  .container {
    width: 100%;
    max-width: 640px;
  }

  .card {
    background: rgba(255, 255, 255, 0.06);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 24px;
    padding: 2.5rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
  }

  .logo {
    text-align: center;
    margin-bottom: 2rem;
  }
  .logo h1 {
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .logo p {
    font-size: 0.9rem;
    color: #888;
    margin-top: 0.3rem;
  }

  label {
    display: block;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #aaa;
    margin-bottom: 0.5rem;
  }

  textarea {
    width: 100%;
    min-height: 120px;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 14px;
    padding: 1rem;
    color: #f0f0f0;
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    resize: vertical;
    transition: border-color 0.2s;
  }
  textarea:focus {
    outline: none;
    border-color: #a78bfa;
  }

  .controls {
    display: flex;
    gap: 1rem;
    margin-top: 1.2rem;
    align-items: end;
  }
  .control-group { flex: 1; }

  select, input[type="range"] {
    width: 100%;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 10px;
    padding: 0.6rem 0.8rem;
    color: #f0f0f0;
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem;
  }
  select option { background: #1a1a2e; }

  input[type="range"] {
    -webkit-appearance: none;
    height: 6px;
    border-radius: 3px;
    background: rgba(255, 255, 255, 0.15);
    border: none;
    padding: 0;
    margin-top: 0.5rem;
  }
  input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 18px; height: 18px;
    border-radius: 50%;
    background: #a78bfa;
    cursor: pointer;
  }

  .speed-val {
    font-size: 0.85rem;
    color: #a78bfa;
    text-align: right;
    margin-top: 0.3rem;
  }

  .btn {
    width: 100%;
    margin-top: 1.5rem;
    padding: 0.9rem;
    border: none;
    border-radius: 14px;
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.25s ease;
    background: linear-gradient(135deg, #7c3aed, #3b82f6);
    color: #fff;
    letter-spacing: 0.02em;
  }
  .btn:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(124, 58, 237, 0.4); }
  .btn:active { transform: translateY(0); }
  .btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    transform: none;
    box-shadow: none;
  }

  .result {
    margin-top: 1.5rem;
    text-align: center;
    min-height: 60px;
  }
  .result audio {
    width: 100%;
    margin-top: 0.5rem;
    border-radius: 10px;
  }

  .status {
    font-size: 0.9rem;
    padding: 0.8rem;
    border-radius: 10px;
    text-align: center;
  }
  .status.loading {
    background: rgba(167, 139, 250, 0.1);
    color: #a78bfa;
    animation: pulse 1.5s ease-in-out infinite;
  }
  .status.error {
    background: rgba(239, 68, 68, 0.1);
    color: #ef4444;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }

  .examples {
    margin-top: 1.5rem;
    padding-top: 1.2rem;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
  }
  .examples p {
    font-size: 0.75rem;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.6rem;
  }
  .example-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
  }
  .chip {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 0.4rem 0.9rem;
    font-size: 0.82rem;
    color: #bbb;
    cursor: pointer;
    transition: all 0.2s;
  }
  .chip:hover {
    background: rgba(167, 139, 250, 0.12);
    border-color: rgba(167, 139, 250, 0.3);
    color: #e0d4ff;
  }
</style>
</head>
<body>
<div class="container">
  <div class="card">
    <div class="logo">
      <h1>🗣️ Bantu TTS</h1>
      <p>Shona Text-to-Speech — powered by FastPitch + HiFi-GAN</p>
    </div>

    <label for="text-input">Enter Shona text</label>
    <textarea id="text-input" placeholder="Ndiri kunzwa tsitsi..."></textarea>

    <div class="controls">
      <div class="control-group">
        <label for="language">Language</label>
        <select id="language">
          <option value="shona" selected>Shona</option>
        </select>
      </div>
      <div class="control-group">
        <label for="speed">Speed</label>
        <input type="range" id="speed" min="0.5" max="1.5" step="0.1" value="1.0">
        <div class="speed-val" id="speed-label">1.0×</div>
      </div>
    </div>

    <button class="btn" id="synth-btn" onclick="synthesize()">
      🎧 Synthesize Speech
    </button>

    <div class="result" id="result"></div>

    <div class="examples">
      <p>Try these examples</p>
      <div class="example-chips">
        <span class="chip" onclick="setExample(this)">Mhoro, makadii?</span>
        <span class="chip" onclick="setExample(this)">Zuva rakanaka nhasi.</span>
        <span class="chip" onclick="setExample(this)">Ndinokuda zvikuru.</span>
        <span class="chip" onclick="setExample(this)">Tinotenda Mwari wedu.</span>
        <span class="chip" onclick="setExample(this)">Zimbabwe inyika yakanaka chaizvo.</span>
        <span class="chip" onclick="setExample(this)">Mwana akadzidza kunyora tsamba.</span>
      </div>
    </div>
  </div>
</div>

<script>
  const speedSlider = document.getElementById('speed');
  const speedLabel = document.getElementById('speed-label');
  speedSlider.addEventListener('input', () => {
    speedLabel.textContent = parseFloat(speedSlider.value).toFixed(1) + '×';
  });

  function setExample(chip) {
    document.getElementById('text-input').value = chip.textContent;
  }

  async function synthesize() {
    const text = document.getElementById('text-input').value.trim();
    if (!text) return;

    const btn = document.getElementById('synth-btn');
    const result = document.getElementById('result');
    btn.disabled = true;
    btn.textContent = '⏳ Generating...';
    result.innerHTML = '<div class="status loading">Synthesizing speech… this may take a moment</div>';

    try {
      const resp = await fetch('/synthesize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: text,
          language: document.getElementById('language').value,
          speed: parseFloat(speedSlider.value),
        }),
      });

      if (!resp.ok) {
        const err = await resp.json();
        throw new Error(err.detail || 'Synthesis failed');
      }

      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      result.innerHTML = '<audio controls autoplay src="' + url + '"></audio>';
    } catch (e) {
      result.innerHTML = '<div class="status error">❌ ' + e.message + '</div>';
    } finally {
      btn.disabled = false;
      btn.textContent = '🎧 Synthesize Speech';
    }
  }

  // Ctrl+Enter shortcut
  document.getElementById('text-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) synthesize();
  });
</script>
</body>
</html>"""
