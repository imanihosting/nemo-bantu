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

## Stack Fitness Assessment (2026-04-19)

A full assessment of whether the chosen stack can deliver the commercial multi-tenant goal lives at `~/.claude/plans/hi-there-kindly-assist-elegant-forest.md`.

**Verdict:** The neural stack (NeMo + FastPitch + HiFi-GAN) is the right choice and is committed to. The surrounding execution (G2P frontend, tokenizer, multi-tenant infra) is not yet fit for commercial launch covering Shona, Ndebele, Zulu, Xhosa.

**Three structural gaps drive the revised remaining phases below:**

1. **Tokenizer + G2P will not survive Zulu/Xhosa.** `BaseCharsTokenizer` in `configs/training/fastpitch_shona.yaml` plus 2-entry stub lexicons in `frontend/lexicons/` cannot represent click consonants (ǀ ǁ ǂ ǃ). Must move to phoneme tokenizer + real lexicons before training Nguni languages.
2. **The G2P "frontend" is two functions, not a frontend.** `frontend/g2p.py` only preserves `mb/nd/ng` clusters; everything else is character splitting. No code-switching, no tone marking, no override dictionaries. PRD's own thesis ("frontend = moat") is contradicted by the current code.
3. **The product layer is single-tenant.** `api/main.py` has no auth, rate limiting, tenant model, observability, or billing. None exist for a commercial multi-tenant launch.

## Remaining Phases (Revised Plan)

### Phase A: Finish Shona on Current Stack (Active → ~2–4 weeks)
- Owner: ML Engineer
- Target Date: 2026-05-10
- Status: In Progress (FastPitch training)
- Let FastPitch training converge on current `BaseCharsTokenizer` setup.
- Train HiFi-GAN vocoder.
- Export `.nemo` checkpoints to `models/fastpitch.nemo` and `models/hifigan.nemo`.
- Run end-to-end synthesis through `inference/synthesize.py` and validate against the API.
- **Purpose: prove the pipeline works end-to-end before investing in frontend overhaul.** Treat this Shona voice as a baseline, not the production voice.

### Phase B: Rebuild the Frontend (4–6 weeks, parallelizable with Phase A)
- Owner: Language Engineer
- Target Date: 2026-06-15
- Status: Foundation Shipped (2026-04-19) — see Session Log below
- Replace `BaseCharsTokenizer` with phoneme-aware tokenizer (NeMo `IPATokenizer` or custom `BantuPhonemeTokenizer`).
- Restructure `frontend/` into per-language modules (`frontend/shona/`, `frontend/ndebele/`, `frontend/zulu/`, `frontend/xhosa/`).
- Build 5k–10k headword lexicons per language with override dictionaries.
- Encode click consonants (ǀ ǁ ǃ) for Zulu/Xhosa with aspiration/voicing/nasalization variants.
- Add Bantu numeral expansion, abbreviation handling, tone markers, code-switching (English insertion) detection.
- Stand up native-speaker validation harness (held-out sentences synthesized on every checkpoint).

### Phase C: Retrain Shona on Phonemes, Then Train Nguni Languages (~6–8 weeks)
- Owner: ML Engineer + Language Engineer
- Target Date: 2026-08-10
- Status: Not Started
- Retrain Shona FastPitch with phoneme inputs to validate Phase B produces measurable quality gain over Phase A baseline.
- Train Ndebele (closest to Shona, fastest second voice).
- Train Zulu, then Xhosa (clicks force Phase B to be done first).
- Reintroduce MFA alignment per language for diagnosability and faster convergence; keep `learn_alignment=true` as fallback.

### Phase D: Commercial Hardening (~4 weeks, parallel with Phase C)
- Owner: Backend/DevOps Engineer
- Target Date: 2026-08-10
- Status: Foundation Shipped (2026-04-19) — see Session Log below
- API auth (API keys or OAuth), per-tenant rate limits (Redis token bucket).
- Tenant model: `tenant_id` on every request, per-tenant voice access control.
- Structured logging (request ID + tenant + voice + char count) and Prometheus metrics.
- Voice catalog endpoint and permissions table.
- Refactor to multi-speaker FastPitch per language + shared HiFi-GAN per sample rate (avoid one-model-per-voice memory blow-up).
- Load testing, billing event emission, basic SLA monitoring.

### Phase E: PRD V1/V2 Expansion (Post-Launch)
- Owner: Product + Engineering Lead
- Target Date: 2026-09-30
- Status: Not Started
- V1: multiple voices per language, emotion control (pitch/speed), batch synthesis.
- V2: full code-switching support, accent control, internal voice cloning workflow.

## Immediate Next Actions

1. Continue monitoring FastPitch training loss in `phase3_fastpitch_training.log` (Phase A).
2. After convergence: train HiFi-GAN vocoder; export checkpoints to `models/`.
3. Begin Phase B in parallel — design phoneme tokenizer interface and start Shona lexicon expansion.
4. Run baseline Shona synthesis through API once checkpoints exist; capture as Phase A reference audio for comparison against Phase C retrained voice.

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
- Phase B (frontend rebuild) can begin in parallel; Phase C (Nguni training) is gated on Phase B.

## Session Log (2026-04-19)

- Performed full stack fitness assessment against the commercial multi-tenant goal across all four target languages (Shona, Ndebele, Zulu, Xhosa).
- Confirmed NeMo + FastPitch + HiFi-GAN core stack is the right choice; no need to evaluate alternatives.
- Identified three structural gaps blocking commercial launch:
  1. `BaseCharsTokenizer` + 2-entry stub lexicons cannot represent Zulu/Xhosa click consonants.
  2. `frontend/g2p.py` and `frontend/normalizer.py` are stubs, not a real frontend; PRD's "frontend = moat" thesis is currently violated by the code.
  3. `api/main.py` is single-tenant — no auth, rate limiting, tenant model, observability, or billing.
- Restructured remaining phases into A (finish Shona baseline), B (rebuild frontend), C (retrain Shona on phonemes + train Nguni), D (commercial hardening), E (PRD V1/V2). B and D are parallelizable with A and C respectively.
- Full assessment saved at `~/.claude/plans/hi-there-kindly-assist-elegant-forest.md`.

### Phase B Foundation Shipped (2026-04-19, parallel with Phase A training)

Frontend rebuild — gap #1 + #2 closed at the architecture level. Native-speaker review and lexicon expansion still required before Phase C training.

- **Per-language package layout** (`frontend/<lang>/`): split monolithic `frontend/g2p.py` and `frontend/normalizer.py` into language-specific modules under `frontend/shona/`, `frontend/ndebele/`, `frontend/zulu/`, `frontend/xhosa/`. Old paths kept as routing shims for backward compatibility.
- **Phoneme inventories with IPA + click consonants** (`frontend/base/phonemes.py`, `frontend/base/nguni.py`, `frontend/<lang>/phonemes.py`): full Nguni click set encoded — dental ǀ, alveolar ǃ, lateral ǁ × five manners (plain, aspirated, voiced, nasal, breathy-voiced nasal). Shona inventory keeps whistled fricatives sv/zv. Greedy longest-match scanner ensures trigraphs (`ngc`) beat digraphs (`nc`) beat single chars (`c`).
- **NeMo-compatible phoneme tokenizer** (`frontend/nemo_tokenizer.py`): `BantuPhonemeTokenizer` with stable IDs across runs, special tokens at low IDs (pad_id=0). Wired into new training config `configs/training/fastpitch_shona_phoneme.yaml` for Phase C; the running Phase A config (`fastpitch_shona.yaml`) is untouched.
- **G2P with lexicon-first + rule-based fallback** (`frontend/base/g2p.py`): override dict → lexicon file → greedy grapheme scan. Returns `G2PResult` with per-word source attribution for native-speaker audit.
- **Normalizer with per-language numerals** (`frontend/base/normalizer.py`, `frontend/<lang>/normalizer.py`): digits 0–10 in Shona/Ndebele/Zulu/Xhosa. Larger ranges and abbreviation handling deferred to native-speaker review queue.
- **Code-switching detector** (`frontend/codeswitch.py`): handles bare tokens, hyphenated compounds, AND prefix-fused English loanwords like `ishopping` / `eshopping` / `kuoffice` (the dominant Nguni pattern; hyphenation is non-native).
- **Lexicon bootstrap** (`frontend/lexicons/<lang>.txt`): expanded from 2 entries per language to ~20–25 high-frequency words each (greetings, pronouns, click-bearing words, common verbs). All entries IPA-encoded and flagged for native-speaker review.
- **Native-speaker validation harness** (`scripts/build_validation_set.py`, `scripts/run_validation_synthesis.py`): emits per-language phonetically-diverse prompt sets covering greetings, prenasalised stops, aspirated stops, clicks, code-switching, numerals. Synthesis script writes `validation_runs/<run_id>/<language>/` with audio + grading CSV form.
- **Test suite expanded** from 6 → 26 tests; all passing. New tests cover prenasalised preservation, whistled fricatives, click consonants per language, aspirated-vs-plain click distinction, trigraph precedence, code-switch detection (prefix-fused + acronym), tokenizer round-trip with clicks, vocab coverage, and pad_id stability.

### Phase D Foundation Shipped (2026-04-19)

Commercial multi-tenant scaffolding — gap #3 closed at the API layer. Production deployment still requires Postgres-backed tenant store and Redis-backed rate limiter.

- **Tenant store** (`api/tenants.py`): in-memory store loaded from `API_KEYS` env (`key:tenant:rate_per_min`). Dev-mode fallback provisions a single `dev` tenant with key `dev-key` so local development and tests work without configuration.
- **API key auth** (`api/auth.py`): FastAPI dependency `require_tenant` validates `X-API-Key` header and returns the resolved `Tenant`. 401 on missing/invalid key.
- **Per-tenant rate limiting** (`api/ratelimit.py`): in-memory token bucket per tenant, capacity = `rate_per_min`, smooth refill. 429 with `Retry-After` header on exhaustion.
- **Structured JSON logging** (`api/logging.py`): per-request `request_id` (returned as `X-Request-ID`), tenant_id binding, latency_ms, status. JSON formatter ready for downstream shipping.
- **Voice catalog** (`api/voices.py`, `GET /voices`): bootstrap catalog with one female voice per supported language; tenant-scoped permissions hook reserved.
- **Updated synthesize endpoint** (`api/main.py`): now requires auth, consumes rate-limit budget, binds tenant to log context. `/health` remains unauthenticated.

### Where We Are Now

- FastPitch Phase A training continues untouched on the full Shona dataset; the running config (`configs/training/fastpitch_shona.yaml`) was not modified.
- Phase B frontend foundation in place — all four languages can now produce IPA phoneme strings including Nguni clicks. Lexicons need ~5k–10k headword expansion per language (native-speaker work).
- Phase C is unblocked: `configs/training/fastpitch_shona_phoneme.yaml` is ready; needs a phonemised manifest (run `text_to_phonemes` over the existing Shona manifest) before training kicks off.
- Phase D foundation in place — API now has auth, rate limiting, structured logging, request IDs, voices catalog. Production needs Postgres + Redis swap and billing event emission.
- Run tests: `python -m pytest tests/ -v` (26 tests, all passing).

### Immediate Next Actions (Updated)

1. Continue monitoring Phase A FastPitch training (`tail -f phase3_fastpitch_training.log`).
2. Begin native-speaker review of lexicon entries in `frontend/lexicons/*.txt` and validation prompts in `scripts/build_validation_set.py`. Add authentic Xhosa code-switching example (TODO marker present).
3. Once Phase A produces a checkpoint, run validation harness against it: `python scripts/build_validation_set.py && python scripts/run_validation_synthesis.py --languages shona`.
4. Build phonemisation step into `training/prepare_data.py` (add `--phonemise` flag) so Phase C manifests can be generated.
5. Expand Shona lexicon toward 5k headwords for the Phase C retraining.
