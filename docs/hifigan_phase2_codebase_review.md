# HiFi-GAN Phase 2 Codebase Review

**Date:** 2026-04-27  
**Reviewer:** Codex  
**Related incident note:** [hifigan_phase2_fix.md](./hifigan_phase2_fix.md)  
**Scope:** Review the current Shona/Bantu TTS codebase, validate the Phase 2 HiFi-GAN fixes, and document follow-up improvements without editing the original fix document.

---

## 1. What This Repo Is Building

The repository is a Shona-first Bantu text-to-speech stack built around NeMo:

1. Raw Shona audio/text is prepared into JSONL manifests.
2. FastPitch is trained as the acoustic model: text to mel spectrogram.
3. HiFi-GAN Phase 1 is fine-tuned as a vocoder on ground-truth mels.
4. HiFi-GAN Phase 2 fine-tunes the vocoder on FastPitch-generated mels paired with ground-truth audio.
5. The API/inference path loads FastPitch and HiFi-GAN checkpoints and returns WAV output.

The core goal of Phase 2 is sound quality: inference uses FastPitch-generated mels, so the vocoder should see that same kind of mel during fine-tuning instead of only pristine ground-truth mels.

---

## 2. Files Reviewed

| File | Role |
|---|---|
| [docs/hifigan_phase2_fix.md](./hifigan_phase2_fix.md) | Existing incident diagnosis and fix notes. Reviewed only; not edited. |
| [training/train_hifigan_phase2.py](../training/train_hifigan_phase2.py) | Phase 2 dataset, trainer monkey patches, checkpoint loading, and dataloaders. |
| [scripts/train_hifigan_phase2_safe.sh](../scripts/train_hifigan_phase2_safe.sh) | tmux launcher and progress/logging wrapper for Phase 2. |
| [scripts/generate_fp_mels.py](../scripts/generate_fp_mels.py) | Pre-generates FastPitch mels used by Phase 2. |
| [configs/training/hifigan_shona.yaml](../configs/training/hifigan_shona.yaml) | HiFi-GAN audio/preprocessor/generator settings. |
| [configs/training/fastpitch_shona.yaml](../configs/training/fastpitch_shona.yaml) | FastPitch audio/model settings. |
| [training/train_hifigan.py](../training/train_hifigan.py) | Phase 1 HiFi-GAN trainer used for comparison. |
| [inference/synthesize.py](../inference/synthesize.py) | Runtime API synthesis checkpoint selection. |
| [scripts/diagnose_quality.py](../scripts/diagnose_quality.py) | Manual quality comparison script. |
| [requirements.txt](../requirements.txt) | Python dependency capture. |

---

## 3. Phase 2 Fix Assessment

### 3.1 Paired crop fix

**Verdict: correct and important.**

The current `FastPitchMelDataset` now does the right high-impact thing:

- Loads the FastPitch mel.
- Computes `n_mel_segments = n_samples // hop_length`.
- Crops the mel to 32 frames when `n_samples=8192` and `hop_length=256`.
- Crops audio from the matching sample offset: `audio_start = mel_start * hop_length`.
- Returns fixed-size tensors that the collate function can stack directly.

That matches the HiFi-GAN architecture in [configs/training/hifigan_shona.yaml](../configs/training/hifigan_shona.yaml):

- `n_window_stride: 256`
- `upsample_rates: [8, 8, 2, 2]`
- Product of upsample rates is `256`

So a 32-frame mel segment maps to `32 * 256 = 8192` generated audio samples.

### 3.2 Runtime dependency fix

**Verdict: correct, but not yet reproducible from the repo.**

The documented `torchaudio.load` crash is real. The referenced log contains:

```text
ImportError: TorchCodec is required for load_with_torchcodec.
```

Installing `torchcodec` plus system `ffmpeg` is a valid fix for this environment. The gap is that [requirements.txt](../requirements.txt) does not list `torchcodec`, and the launcher does not preflight-check `ffmpeg`, so a fresh setup can hit the same failure again.

### 3.3 Launcher progress fix

**Verdict: good operational fix.**

The Phase 2 launcher now runs the trainer under:

```bash
script -q -f -e -a -c "<python command>" "$LOGFILE"
```

That preserves a pseudo-TTY, which lets Lightning/Rich render live progress. The `-e` flag is also important because it preserves the child command exit code.

### 3.4 Orphan cleanup fix

**Verdict: useful, but broad.**

The added:

```bash
pkill -f "train_hifigan_phase2.py"
```

addresses the observed orphan process leak from `script`/tmux relaunching. It is effective for the current workstation workflow, but it can kill any other process whose command line contains `train_hifigan_phase2.py`.

Longer term, track the launched child PID or process group and terminate only that process tree.

---

## 4. Highest-Priority Findings

| Priority | Finding | Why It Matters | Recommended Fix |
|---|---|---|---|
| P0 | Inference and diagnosis still point at Phase 1 HiFi-GAN checkpoints. | Phase 2 may train successfully, but the app can still serve the Phase 1 vocoder. That makes the quality gain invisible. | Update [inference/synthesize.py](../inference/synthesize.py) and [scripts/diagnose_quality.py](../scripts/diagnose_quality.py) to prefer Phase 2 checkpoints under `outputs/hifigan_shona_phase2/HiFiGAN_Shona_Phase2`, then fall back to Phase 1. |
| P1 | `torchcodec` and `ffmpeg` are operational requirements but not encoded in setup. | Clean installs can reproduce the original startup crash. | Add `torchcodec` to Python dependencies, document/install system `ffmpeg`, and add launcher preflight checks. |
| P1 | FastPitch-generated mels may not be perfectly aligned with ground-truth audio. | Phase 2 trains the vocoder against recorded audio, but `generate_fp_mels.py` uses FastPitch predicted durations/prosody. If those differ from the recording, the target audio can be temporally mismatched. | Measure generated mel length versus ground-truth mel length. Consider teacher-forced FastPitch mels using ground-truth duration/pitch/alignment where NeMo exposes that path. |
| P1 | Validation crops are random. | `shuffle=False` does not make validation deterministic because `__getitem__` still uses `random.randint`. Checkpoint selection by `val_loss` can be noisy. | Add a deterministic validation mode: center crop, fixed crop, or stable per-sample crop offset. |
| P1 | Mel generation can silently reuse stale mels. | If FastPitch is retrained, existing `.pt` files are reused without checking which checkpoint produced them. | Add `--force`, checkpoint metadata, config metadata, and a manifest generation summary. |
| P1 | Mel generation hides failures. | `except Exception` increments `skipped` but does not log the sample path or exception. | Log every skipped item and exception to a sidecar failure report. |
| P2 | Mel filenames use only `Path(audio_path).stem`. | Duplicate basenames from different folders can collide. | Prefix with manifest index or a short hash of the full audio path. |
| P2 | Phase 2 trainer hardcodes config-sensitive constants. | `sample_rate`, `hop_length`, `n_samples`, batch size, and worker counts can drift from YAML config. | Derive these from `hifigan_shona.yaml` and assert consistency. |
| P2 | Global `torch.load` monkey patch sets `weights_only=False`. | It unblocks NeMo checkpoints, but it changes every later `torch.load` call and broadens pickle execution risk. | Replace with local helpers for trusted checkpoints and keep mel loads explicit. |
| P2 | Phase 2 launcher assumes `logs/` exists. | Clean checkout launches can fail before useful training diagnostics are written. | Add `mkdir -p "$PROJECT_DIR/logs"` near the top of the launcher. |

---

## 5. Recommended Next Implementation Order

1. **Make Phase 2 usable by inference.**  
   Add shared checkpoint discovery that prefers Phase 2 HiFi-GAN checkpoints and falls back to Phase 1.

2. **Make the environment reproducible.**  
   Add `torchcodec`, document/install `ffmpeg`, and fail fast with clear preflight messages.

3. **Make validation deterministic.**  
   Training should keep random crops; validation should use stable crops so `val_loss` means something.

4. **Make generated mels traceable.**  
   Store metadata for the FastPitch checkpoint/config that produced each mel manifest.

5. **Reduce hardcoded Phase 2 assumptions.**  
   Read sample rate, hop length, mel channels, segment size, and upsample product from config and assert them.

6. **Tighten process cleanup.**  
   Replace broad `pkill -f` with PID/process-group cleanup.

---

## 6. Suggested Acceptance Checks

### 6.1 Dependency preflight

```bash
.venv/bin/python -c "import torchcodec, torchaudio; print('audio stack OK')"
ffmpeg -version
```

### 6.2 Dataset shape check

```bash
.venv/bin/python -c "from training.train_hifigan_phase2 import FastPitchMelDataset, collate_fn; ds=FastPitchMelDataset('data/manifests/shona_fp_mels_val.jsonl', n_samples=8192, hop_length=256); a, al, m, ml = collate_fn([ds[0], ds[1]]); assert a.shape == (2, 8192); assert m.shape == (2, 80, 32); print('paired crop OK')"
```

### 6.3 Checkpoint selection check

After Phase 2 produces at least one checkpoint, inference should explicitly report which vocoder checkpoint was selected:

```bash
.venv/bin/python -c "from inference.synthesize import NemoSynthesizer; s=NemoSynthesizer.get_instance(); print('loaded:', s.is_loaded)"
```

Expected behavior:

- Prefer a Phase 2 checkpoint when present.
- Fall back to Phase 1 only when no Phase 2 checkpoint exists.
- Print or log the selected checkpoint path.

### 6.4 Mel freshness check

Regenerating mels after a new FastPitch checkpoint should either:

- refuse to reuse stale files unless `--force` is passed, or
- write a new manifest/output directory keyed by checkpoint identity.

---

## 7. Notes From Log Review

The existing fix document's performance numbers are supported by the current training log:

- `logs/hifigan_phase2_20260427_144748.log` shows `train_step_timing` around `0.407-0.436s`.
- The same log shows about `2.42it/s` and epoch ETA around `9-10 min`.
- `logs/hifigan_phase2_20260427_113026.log` contains the TorchCodec import failure.

These observations support the conclusion that the paired crop fix solved the major performance issue.

---

## 8. Summary

The Phase 2 crop fix is the right fix for the huge training slowdown. It changes the workload from full-utterance vocoding with most output discarded to normal HiFi-GAN segment training. The current implementation now matches the intended tensor geometry.

The biggest remaining risk is not speed; it is integration. The rest of the project needs to reliably consume Phase 2 checkpoints, reproduce the dependency fix, and validate with deterministic metrics. Once those are handled, Phase 2 becomes a repeatable part of the training pipeline instead of a successful one-off rescue.
