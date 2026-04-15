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

- Project is at **Phase 3: Training Execution — IN PROGRESS**.
- FastPitch training is actively running on 12,954 Shona samples.
- MFA alignment skipped in favor of `learn_alignment=true` (NeMo internal alignment).

## Phase 3: Training Execution (Active)

### Resolved Blockers
- **nv_one_logger**: Resolved by creating stub package in venv (NVIDIA-internal telemetry not available on PyPI).
- **MFA _kalpy**: Bypassed by using `learn_alignment=true` in FastPitch config (MFA is optional).
- **NeMo import chain**: Installed all transitive deps (`einops`, `transformers`, `sentencepiece`, `kaldialign`, `pyannote.core`, `pyannote.metrics`, `jiwer`, `ipython`).
- **Training script architecture**: Rewrote `train_fastpitch.py` and `train_hifigan.py` to use NeMo's official Hydra + PyTorch Lightning pattern instead of broken `python -m` subprocess calls.

### Active Training
- **Model**: FastPitch (45.7M params) with AlignmentEncoder
- **Config**: `configs/training/fastpitch_shona.yaml`
- **Dataset**: 12,954 Shona wav/txt pairs from `badrex/shona-speech`
- **Tokenizer**: Character-level (`BaseCharsTokenizer` — Shona is phonemic)
- **Epochs**: 1000 target
- **Log**: `phase3_fastpitch_training.log`
- **Checkpoints**: `outputs/fastpitch_shona/FastPitch_Shona/checkpoints/`

### Pending
- Monitor FastPitch training loss convergence
- After FastPitch converges: train HiFi-GAN vocoder (`python training/train_hifigan.py`)
- Export final `.nemo` checkpoints to `models/`

## Remaining Phases

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

1. Monitor FastPitch training loss in `phase3_fastpitch_training.log`.
2. After convergence (~200-500 epochs), train HiFi-GAN vocoder.
3. Export trained checkpoints to `models/fastpitch.nemo` and `models/hifigan.nemo`.
4. Run native-speaker pronunciation validation loop.

## Session Log (2026-04-15)

- Cloned nemo-bantu repo to workstation.
- Fetched full Shona dataset from HuggingFace (badrex/shona-speech): 12,954 wav/txt pairs, 12GB.
- Installed `datasets<3.0` (compatible audio backend via soundfile/librosa instead of torchcodec).
- Prepared and validated NeMo manifest (12,954 entries).
- Created `nv_one_logger` stub package to unblock NeMo 2.7.2 imports.
- Installed all NeMo transitive dependencies (einops, transformers, sentencepiece, kaldialign, pyannote, jiwer, ipython).
- Created production-ready configs:
  - `configs/training/fastpitch_shona.yaml` (character tokenizer, learn_alignment=true)
  - `configs/training/hifigan_shona.yaml` (v1 generator, self-contained)
- Rewrote training scripts using NeMo's official Hydra + Lightning pattern.
- Updated `scripts/phase3_run.sh` to default SKIP_MFA=1.
- Updated `.gitignore` to exclude data/raw, data/processed, models/*.nemo.
- Ran successful 1-epoch smoke test (val_loss=54.46, checkpoint saved).
- Launched full 1000-epoch FastPitch training in background.

### Where We Are Now

- FastPitch training is **actively running** on the full Shona dataset.
- Monitor with: `tail -f phase3_fastpitch_training.log`
- Check checkpoint dir: `ls outputs/fastpitch_shona/FastPitch_Shona/checkpoints/`
- Phase 4 starts after first successful trained artifact is produced.
