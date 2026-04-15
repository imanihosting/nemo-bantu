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
- Owner: ML Engineer
- Target Date: 2026-05-15
- Status: In Progress
- Install and pin exact NeMo/MFA environment versions.
- Prepare clean manifests with real durations and validated transcripts.
- Run MFA alignment for each language dataset.
- Train FastPitch and HiFi-GAN per target voice.
- Save/checkpoint model artifacts under `models/`.

### Phase 4: Pronunciation and Frontend Accuracy
- Owner: Language Engineer
- Target Date: 2026-06-05
- Status: Not Started
- Build stronger normalization rules for:
  - numbers, abbreviations, names, slang, and code-switching
- Expand per-language lexicons and override dictionaries.
- Add phoneme inventory controls per language and QA with native speakers.

### Phase 5: Quality and Performance Optimization
- Owner: ML + Platform Team
- Target Date: 2026-06-25
- Status: Not Started
- Evaluate:
  - pronunciation accuracy
  - MOS/naturalness
  - latency target (<500ms where feasible)
  - artifact/stability checks
- Tune inference speed and memory footprint.
- Add caching/batch strategies for production workloads.

### Phase 6: Production Readiness
- Owner: Backend/DevOps Engineer
- Target Date: 2026-07-10
- Status: Not Started
- Add observability:
  - request metrics, latency tracking, error logging
- Add deployment packaging:
  - service configuration, startup scripts, environment profiles
- Harden API behavior for multi-voice and multi-language usage.
- Add load testing and failure recovery checks.

### Phase 7: PRD Roadmap Expansion (V1 and V2)
- Owner: Product + Engineering Lead
- Target Date: 2026-08-15
- Status: Not Started
- V1:
  - multiple voices
  - emotion control (pitch/speed)
  - batch synthesis
- V2:
  - code-switching support
  - accent control
  - internal voice cloning workflow

## Immediate Next Actions

1. Pull Shona source data from Hugging Face (`badrex/shona-speech`) using `python scripts/fetch_shona_hf.py`.
2. Run `./scripts/phase3_run.sh shona` to execute manifest -> validation -> alignment -> training entrypoints.
3. Replace placeholder checkpoints with trained artifacts in `models/`.
4. Run native-speaker pronunciation validation loop.

## Session Log (2026-04-15)

- Switched to remote-only workflow on workstation path /home/blaquesoul/Desktop/nemo-bantu.
- Verified Shona dataset availability under data/raw/shona (12,954 wav/txt pairs used in manifest build).
- Verified MFA binary exists in project venv (Montreal_Forced_Aligner 3.3.9), but runtime import remains blocked by missing _kalpy.
- Added and validated Phase-3 script controls on remote:
  - scripts/phase3_run.sh supports SKIP_MFA=1
  - training/align_mfa.py supports CLI args (--corpus-dir, --dictionary, --acoustic-model, --output-dir)
- Ran Phase-3 baseline in background with SKIP_MFA=1:
  - Manifest generation: success (entries=12954)
  - Manifest validation: success (entries=12954)
  - MFA: intentionally skipped
  - FastPitch training start: failed before training due NeMo import chain
- Current training blocker:
  - ModuleNotFoundError: No module named nv_one_logger from NeMo Lightning callback path.
- Current MFA blocker:
  - ModuleNotFoundError: No module named _kalpy.

### Where We Left Off

- Phase 3 remains In Progress.
- Data preparation is working and repeatable.
- Training/alignment runtime environments still need final compatibility fixes (nv_one_logger, _kalpy).
- Phase 4 should start only after one successful Phase-3 end-to-end run (at least one trained artifact).
