# Bantu TTS Platform (NeMo + G2P + FastPitch + HiFi-GAN + MFA)

Production scaffold for a Bantu language TTS system based on the PRD in `bantu_tts_prd.md`.

## Core Stack

- NVIDIA NeMo: https://github.com/NVIDIA/NeMo
- NeMo TTS Docs: https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/tts/intro.html
- FastPitch paper: https://arxiv.org/abs/2006.06873
- HiFi-GAN: https://github.com/jik876/hifi-gan
- MFA Docs: https://montreal-forced-aligner.readthedocs.io/
- MFA GitHub: https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner

## Repository Layout

```text
nemo-bantu/
  api/
  configs/
    languages/
    models/
    training/
  data/
    raw/
    processed/
    manifests/
  frontend/
    lexicons/
  inference/
  scripts/
  training/
  tests/
```

## Quickstart

1. Create and activate virtual environment:
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Copy env file:
   - `cp .env.example .env`
4. Run API:
   - `uvicorn api.main:app --reload --port 8000`
5. Run tests:
   - `pytest -q`

## Pipeline (PRD-aligned)

Text -> Normalization -> G2P -> FastPitch (mel) -> HiFi-GAN (waveform) -> Audio Output

## Current Status

Phase 2 scaffold status:
- API contract is implemented (`POST /synthesize`).
- Frontend normalization and lexicon-first G2P support `shona`, `ndebele`, `zulu`, `xhosa`.
- NeMo inference wrapper is integrated with safe fallback when checkpoints are missing.
- MFA alignment command is wired with validation.
- Training scripts execute NeMo module entrypoints using config files.

## Checkpoint Placement

Place NeMo checkpoints at:
- `models/fastpitch.nemo`
- `models/hifigan.nemo`

When these files exist and `nemo_toolkit` is installed, API synthesis uses NeMo models.

## Phase 3 Execution

Fetch Shona data from Hugging Face:
- `python scripts/fetch_shona_hf.py --dataset badrex/shona-speech --split train --output-dir data/raw/shona`

Run the phase-3 pipeline (example for Shona):
- `chmod +x scripts/phase3_run.sh`
- `./scripts/phase3_run.sh shona`

This will:
- prepare a duration-aware manifest
- validate manifest integrity
- run MFA alignment
- start FastPitch and HiFi-GAN training entrypoints
