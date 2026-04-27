# HiFi-GAN Phase 2 — Diagnosis & Fix

**Date:** 2026-04-27
**Status:** Resolved — training running healthily.
**Outcome:** Step time **70.0s → 0.41s** (170× faster). Epoch time **~28 hrs → ~10 min**. Full 200-epoch run **~234 days → ~33 hrs**.

---

## 1. Summary

The HiFi-GAN Phase 2 training launcher (`scripts/train_hifigan_phase2_safe.sh`) appeared to start, then either crashed silently or ran with throughput so low it would have taken months to finish a single epoch. Three problems were stacked on top of each other:

| # | Problem | Severity | Visible symptom |
|---|---|---|---|
| 1 | `torchaudio.load` had no working backend | **Crash** | Process died ~4 s after start, no checkpoint, no clear error in NeMo log |
| 2 | Launcher piped Python through `tee`, which suppressed the live progress bar | **UX** | tmux pane stuck on the setup banner; no idea if training was alive |
| 3 | Custom `FastPitchMelDataset` cropped audio to 8192 samples but loaded the **full** mel | **Performance** | 70 s/step compute, 96 % of which produced audio that was immediately discarded |

The first two were obvious from the traceback / TTY behaviour. The third was the one that mattered: it explains why even after fixing the crash and dataloader workers, step time refused to drop below ~60 s.

---

## 2. Background — what Phase 2 is supposed to do

Phase 1 trains HiFi-GAN on **ground-truth mels** (audio → mel via the on-the-fly preprocessor → reconstruct). Inference uses **FastPitch-generated mels**, which look slightly different from ground-truth mels. Phase 2 closes that train/inference gap by fine-tuning the vocoder on `(FastPitch mel, ground-truth audio)` pairs. The setup files:

- [scripts/generate_fp_mels.py](../scripts/generate_fp_mels.py) — pre-generates 12,954 FastPitch mels into `data/fastpitch_mels/`.
- [training/train_hifigan_phase2.py](../training/train_hifigan_phase2.py) — defines `FastPitchMelDataset`, monkey-patches HiFi-GAN's `training_step`/`validation_step` to use those paired (mel, audio) tensors, loads the Phase 1 checkpoint, and trains.
- [scripts/train_hifigan_phase2_safe.sh](../scripts/train_hifigan_phase2_safe.sh) — tmux launcher.

---

## 3. Problem 1 — silent crash on startup

### Symptom

`tmux attach -t hifigan_phase2` returned "no server running". `outputs/hifigan_shona_phase2/HiFiGAN_Shona_Phase2/` had only a setup-stage NeMo log; no `checkpoints/` directory; no obvious error.

### Diagnosis

The traceback was tee'd to the launcher log (`logs/hifigan_phase2_20260427_113026.log`) but not to the NeMo log, because the crash fired **before** NeMo's logger reached the training-loop region. End of the launcher log:

```
ImportError: TorchCodec is required for load_with_torchcodec.
  Please install torchcodec to use this function.
```

`torchaudio 2.11.0+cu130` (the installed version) removed its built-in audio backends and now delegates `torchaudio.load()` to `torchcodec`, which itself wraps system FFmpeg. Neither was installed in this environment. The crash fired on the first batch fetch of Lightning's sanity-check val loop — `__getitem__` at [training/train_hifigan_phase2.py:76](../training/train_hifigan_phase2.py#L76).

Phase 1 wasn't affected because it used NeMo's stock `MelAudioDataset`, which loads audio through NeMo's own preprocessor stack and never calls `torchaudio.load`.

### Fix

Install the proper dependency stack (chosen over a soundfile workaround at the user's request — keeps the project on its intended audio path):

```bash
sudo apt-get install -y ffmpeg          # system: ffmpeg 6.1.1
.venv/bin/pip install torchcodec        # venv: torchcodec 0.11.1
```

Verification — `torchaudio.load` now round-trips a manifest WAV:
```
audio torch.Size([1, 491098]) dtype torch.float32 sr 22050
```

No code changes required for this fix.

---

## 4. Problem 2 — invisible progress bar

### Symptom

After fixing the crash, training was running (verified via `nvidia-smi` showing the Python process at ~22 GB GPU memory) but the tmux pane was frozen on the setup banner forever. `tail -f` of the log file showed only the post-setup `train_dataloader does not have many workers` warning, then nothing.

### Diagnosis

The launcher piped Python's stdout through `tee`:

```bash
"$VENV/python" "$PROJECT_DIR/training/train_hifigan_phase2.py" --epochs 200 --batch-size 8 \
    2>&1 | tee -a "$LOGFILE"
```

Lightning's Rich/tqdm progress bar checks `sys.stdout.isatty()` before rendering. Through a `tee` pipe stdout is not a TTY, so the bar is silently downgraded or suppressed entirely. Result: training is running, but you have zero in-band signal until validation epoch 10 (which would happen ~12 days in at the then-current step time).

### Fix

Replace `tee` with `script` in [scripts/train_hifigan_phase2_safe.sh:48](../scripts/train_hifigan_phase2_safe.sh#L48):

```bash
script -q -f -e -a -c "<python ...>" "$LOGFILE"
```

`script` runs the child inside a pseudo-TTY, so:

- `-q` suppresses script's banner.
- `-f` flushes after every write so `tail -f` sees updates instantly.
- `-e` exits with the child's exit code (preserves `set -e` semantics).
- `-a` appends to the existing logfile so the launcher's startup header stays at top.

Now both `tmux attach` and `tail -f` show the live Rich bar with carriage-return updates, and the bar is captured persistently in the log file.

### Side-effect — orphan process leak

`script` runs the trainer in its own pty session. `tmux kill-session -t hifigan_phase2` does not propagate to the Python grandchild through that boundary. After the first relaunch we ended up with two training processes on the GPU at once (45 GB orphan + 11 GB new run, fighting for memory).

Fix in the launcher ([scripts/train_hifigan_phase2_safe.sh:61-63](../scripts/train_hifigan_phase2_safe.sh#L61-L63)):

```bash
tmux kill-session -t "$SESSION" 2>/dev/null || true
pkill -f "train_hifigan_phase2.py" 2>/dev/null || true
sleep 2
```

Now relaunching reliably reclaims the GPU.

---

## 5. Problem 3 (the real one) — 97 % wasted GPU compute

### Symptom

After Problems 1 and 2 were fixed and the dataloader was given workers (`num_workers=8`), GPU utilization went from 50-66 % to 96 %, but **step time only dropped from ~60 s to ~70 s**. Lightning's ETA for one epoch was 28 hours, and 200 epochs would have taken ~234 days. RAM, batch size, and precision were not the bottleneck — the GPU was fully busy doing real compute, just compute that got thrown away.

### Diagnosis

Compare how Phase 1 and Phase 2 set up their training tensors.

**Phase 1** (NeMo's stock `VocoderDataset`, [configs/training/hifigan_shona.yaml:62-71](../configs/training/hifigan_shona.yaml#L62-L71)) crops both audio AND mel to a paired window:

- audio crop: `n_samples = 8192`
- mel crop: `8192 / hop_length(256) = 32 frames`
- Generator runs on 32 mel frames → produces 8192 audio samples → loss vs 8192 target samples. Aligned.

**Phase 2's custom `FastPitchMelDataset`** (the original code, before the fix) only cropped the audio:

```python
# Random crop AUDIO to 8192 samples (= 32 mel-frame equivalent)
if audio.shape[0] >= self.n_samples:
    start = random.randint(0, audio.shape[0] - self.n_samples)
    audio = audio[start:start + self.n_samples]
…
# Load FULL mel (entire utterance, often 800-1900 frames)
mel = torch.load(entry["mel_filepath"], map_location="cpu")
return audio, audio.shape[0], mel, mel.shape[1]
```

Then in `_phase2_training_step`:

```python
audio_pred = self.generator(x=fp_mel)             # generates fp_mel.shape[1] * 256 samples
min_audio_len = min(audio.shape[2], audio_pred.shape[2])  # = 8192
audio = audio[:, :, :min_audio_len]               # keep first 8192
audio_pred = audio_pred[:, :, :min_audio_len]     # discard ~210k samples per item
```

For a typical ~10-second Shona utterance (~860 mel frames), the generator produced **860 × 256 ≈ 220,000 audio samples per item**, of which only **8,192 (3.7 %)** were used. The other 96.3 % was discarded a microsecond after being computed. Multiply by batch size 8 and that's an enormous amount of waste per step. The discriminators (MPD + MSD) similarly evaluated the full prediction.

This explains every previous observation cleanly:
- 70 s/step compute is **genuine** — the generator really is producing 30× more audio than needed.
- 96 % GPU util means the workers do feed it; the work itself is mostly waste.
- Bigger RAM, bigger batch, bf16 would each shave 1.5-2× — band-aids on a 33× wound.
- A proper paired crop should drop step time to ~2 s without touching anything else.

### Fix

Crop the **mel** to a random 32-frame window in `__getitem__`, and crop the audio to the **same** window. Single, contained change to `FastPitchMelDataset`.

**Constructor** ([training/train_hifigan_phase2.py:52-58](../training/train_hifigan_phase2.py#L52-L58)) — add `hop_length` so the dataset knows how mel frames map to audio samples:

```python
def __init__(self, manifest_path: str, sample_rate: int = 22050,
             n_samples: int = 8192, min_duration: float = 0.5,
             hop_length: int = 256):
    self.sample_rate = sample_rate
    self.n_samples = n_samples
    self.hop_length = hop_length
    self.n_mel_segments = n_samples // hop_length  # 32 frames for 8192 samples
    self.entries = []
```

`hop_length=256` is fixed by the HiFi-GAN architecture: `upsample_rates: [8, 8, 2, 2]` → product 256 ([configs/training/hifigan_shona.yaml:55](../configs/training/hifigan_shona.yaml#L55)), same as `n_window_stride` ([configs/training/hifigan_shona.yaml:16](../configs/training/hifigan_shona.yaml#L16)).

**`__getitem__`** ([training/train_hifigan_phase2.py:74-101](../training/train_hifigan_phase2.py#L74-L101)) — paired crop:

```python
def __getitem__(self, idx):
    entry = self.entries[idx]

    audio, sr = torchaudio.load(entry["audio_filepath"])
    if sr != self.sample_rate:
        audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
    audio = audio[0]  # mono, [T]

    mel = torch.load(entry["mel_filepath"], map_location="cpu")  # [n_mel, M]

    # Reconcile audio length to mel.shape[1] * hop_length so paired
    # crop windows always line up (FastPitch mels may not match the
    # recorded audio's exact frame count).
    expected_audio_len = mel.shape[1] * self.hop_length
    if audio.shape[0] > expected_audio_len:
        audio = audio[:expected_audio_len]
    elif audio.shape[0] < expected_audio_len:
        audio = F.pad(audio, (0, expected_audio_len - audio.shape[0]))

    # Paired random crop: same start in mel and audio domains.
    if mel.shape[1] >= self.n_mel_segments:
        mel_start = random.randint(0, mel.shape[1] - self.n_mel_segments)
        mel = mel[:, mel_start:mel_start + self.n_mel_segments]
        audio_start = mel_start * self.hop_length
        audio = audio[audio_start:audio_start + self.n_samples]
    else:
        mel = F.pad(mel, (0, self.n_mel_segments - mel.shape[1]))
        audio = F.pad(audio, (0, self.n_samples - audio.shape[0]))

    return audio, audio.shape[0], mel, mel.shape[1]
```

**`collate_fn`** ([training/train_hifigan_phase2.py:104-112](../training/train_hifigan_phase2.py#L104-L112)) — simplified, since mels are now fixed-size and need no padding:

```python
def collate_fn(batch):
    audios, audio_lens, mels, mel_lens = zip(*batch)
    return (
        torch.stack(audios),
        torch.tensor(audio_lens),
        torch.stack(mels),
        torch.tensor(mel_lens),
    )
```

**Caller** ([training/train_hifigan_phase2.py:284-285](../training/train_hifigan_phase2.py#L284-L285)) — pass `hop_length`:

```python
train_ds = FastPitchMelDataset(train_manifest, n_samples=8192, min_duration=0.5, hop_length=256)
val_ds = FastPitchMelDataset(val_manifest, n_samples=8192, min_duration=0.5, hop_length=256)
```

---

## 6. Verification — measured outcome

Smoke test (run before relaunching):

```bash
.venv/bin/python -c "
from training.train_hifigan_phase2 import FastPitchMelDataset, collate_fn
ds = FastPitchMelDataset('data/manifests/shona_fp_mels_val.jsonl', n_samples=8192, hop_length=256)
batch = collate_fn([ds[0], ds[1], ds[2], ds[3]])
audio, alen, mel, mlen = batch
assert audio.shape == (4, 8192)
assert mel.shape == (4, 80, 32)
print('paired crop OK')
"
# → audio torch.Size([4, 8192]) torch.float32
#   mel   torch.Size([4, 80, 32]) torch.float32
#   paired crop OK
```

Live training, steady-state (Epoch 0, steps 37-46 of [logs/hifigan_phase2_20260427_144748.log](../logs/hifigan_phase2_20260427_144748.log)):

| Metric | Value |
|---|---|
| `train_step_timing` | 0.407–0.436 s (mean ~0.41) |
| Throughput | 2.42 it/s |
| Lightning epoch ETA | 0:09:42 |
| GPU utilization | 93 % |
| `g_l1_loss` | 1.4-1.8 (sane, non-NaN) |

Comparison:

| | Before | After |
|---|---|---|
| Mel frames per item | ~860 (full utterance) | 32 (paired) |
| Generator output samples | ~220,000 | 8,192 |
| Wasted compute per step | ~96 % | 0 % |
| Step time | ~70 s | **0.41 s** |
| Epoch time | ~28 hrs | **~10 min** |
| First checkpoint (epoch 10) | ~12 days | **~1h 40m** |
| Full 200 epochs | ~234 days | **~33 hrs** |

---

## 7. Files changed

| Path | Change |
|---|---|
| [training/train_hifigan_phase2.py](../training/train_hifigan_phase2.py) | Paired mel/audio crop in `__getitem__`; add `hop_length` to `__init__`; simplify `collate_fn`; pass `hop_length=256` from `main()` |
| [scripts/train_hifigan_phase2_safe.sh](../scripts/train_hifigan_phase2_safe.sh) | `tee` → `script -fqea` to preserve TTY; `pkill -f train_hifigan_phase2.py` after `tmux kill-session` to prevent orphan leaks |
| **(env)** `.venv` | Added `torchcodec==0.11.1` |
| **(system)** | Added `ffmpeg 6.1.1` via apt |

---

## 8. Lessons / preventive notes

- **A custom `Dataset` for HiFi-GAN must crop the mel as well as the audio.** The on-the-fly mel preprocessor in NeMo's `VocoderDataset` does this implicitly because it computes the mel from the cropped audio. Anything that supplies a pre-computed mel must do it manually, with the start point shared between mel and audio domains.
- **High GPU util ≠ healthy training.** 96 % utilization can still be 96 % wasted compute. Always verify that the tensor shapes flowing through the model are the shapes you intended.
- **Always keep stdout a TTY through your launcher.** `tee` silently breaks any progress bar that uses `\r` for in-place updates. `script -f` is the cheap fix; pipe-aware tools like `unbuffer` are alternatives.
- **`tmux kill-session` doesn't always kill grandchildren.** When the inner shell launches anything that creates a new pty (`script`, `nohup`, `setsid`), follow up with an explicit `pkill -f <script_basename>`.
- **A clear traceback existed all along** in `logs/hifigan_phase2_<timestamp>.log` — but the user was looking at the NeMo log, which only sees what NeMo's logger has captured. Future launchers should either consolidate logs or echo a "if there's no progress, look in $LOGFILE for traceback" hint at startup.
