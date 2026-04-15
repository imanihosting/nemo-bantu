# Bantu TTS Project Progress

This document tracks what has been implemented so far and what remains, based on `bantu_tts_prd.md`.

## Completed Work

### Phase 1: Foundation Scaffold (Completed)
- Created project structure for:
  - `api/`, `frontend/`, `inference/`, `training/`, `configs/`, `data/`, `scripts/`, `tests/`
- Added environment and setup files:
  - `README.md`, `requirements.txt`, `.env.example`
- Added base configs:
  - `configs/base.yaml`
  - `configs/training/fastpitch_train.yaml`
  - `configs/training/hifigan_train.yaml`
  - `configs/models/fastpitch.yaml`
  - `configs/models/hifigan.yaml`
- Added data staging folders:
  - `data/raw/`, `data/processed/`, `data/manifests/`

### Phase 2: Core Integration and Multilingual Prep (Completed)
- API contract implemented and hardened:
  - `POST /synthesize` in `api/main.py`
  - Validation now returns `400` for unsupported languages
- Frontend pipeline implemented:
  - Text normalization in `frontend/normalizer.py`
  - Lexicon-first G2P in `frontend/g2p.py`
  - Prenasalized cluster preservation in fallback G2P (`mb`, `nd`, `ng`)
- Language packs added:
  - Configs: `shona`, `ndebele`, `zulu`, `xhosa`
  - Lexicons for all 4 languages
- Inference integration path added:
  - `inference/synthesize.py` now attempts NeMo FastPitch + HiFi-GAN loading
  - Graceful fallback if checkpoints are missing
- Training and alignment entrypoints upgraded:
  - `training/train_fastpitch.py`
  - `training/train_hifigan.py`
  - `training/align_mfa.py`
  - `training/prepare_data.py`
- Tests expanded:
  - API tests and frontend G2P behavior tests

## Current Status

- Project is at a **functional phase-2 scaffold state**.
- End-to-end architecture is in place.
- Real model quality depends on real checkpoints, real datasets, and language-specific normalization/G2P refinements.

## Remaining Phases

### Phase 3: Training Execution (Next)
- Install and pin exact NeMo/MFA environment versions.
- Prepare clean manifests with real durations and validated transcripts.
- Run MFA alignment for each language dataset.
- Train FastPitch and HiFi-GAN per target voice.
- Save/checkpoint model artifacts under `models/`.

### Phase 4: Pronunciation and Frontend Accuracy
- Build stronger normalization rules for:
  - numbers, abbreviations, names, slang, and code-switching
- Expand per-language lexicons and override dictionaries.
- Add phoneme inventory controls per language and QA with native speakers.

### Phase 5: Quality and Performance Optimization
- Evaluate:
  - pronunciation accuracy
  - MOS/naturalness
  - latency target (<500ms where feasible)
  - artifact/stability checks
- Tune inference speed and memory footprint.
- Add caching/batch strategies for production workloads.

### Phase 6: Production Readiness
- Add observability:
  - request metrics, latency tracking, error logging
- Add deployment packaging:
  - service configuration, startup scripts, environment profiles
- Harden API behavior for multi-voice and multi-language usage.
- Add load testing and failure recovery checks.

### Phase 7: PRD Roadmap Expansion (V1 and V2)
- V1:
  - multiple voices
  - emotion control (pitch/speed)
  - batch synthesis
- V2:
  - code-switching support
  - accent control
  - internal voice cloning workflow

## Immediate Next Actions

1. Install dependencies and run tests.
2. Place real NeMo checkpoints in `models/`.
3. Start first language full training cycle (Shona recommended).
4. Run native-speaker pronunciation validation loop.
