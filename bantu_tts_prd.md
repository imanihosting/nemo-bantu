# Bantu TTS Platform – Production PRD
## (NeMo + Custom G2P + FastPitch + HiFi-GAN + MFA)

---

## 1. Overview

**Goal:**  
Build a proprietary, production-grade Text-to-Speech (TTS) system optimized for Bantu languages (e.g., Shona, Ndebele, Zulu, Xhosa).

**Core Principle:**  
Language authenticity comes from **frontend (G2P + normalization)**, not just the neural model.

---

## 2. Objectives

- Build **fully proprietary TTS voices**
- Achieve **native-level pronunciation accuracy**
- Enable **fast training cycles (days, not months)**
- Support **multi-language African expansion**
- Maintain **production-grade performance**

---

## 3. Tech Stack

### Core
- NVIDIA NeMo (training framework)
- FastPitch (acoustic model)
- HiFi-GAN (vocoder)
- Montreal Forced Aligner (MFA)

### Frontend (Critical)
- Custom G2P (Grapheme-to-Phoneme)
- Text normalization engine
- Pronunciation lexicons per language

---

## 4. System Architecture

### Pipeline

Text Input  
→ Normalization (numbers, abbreviations)  
→ G2P Conversion (custom phoneme rules)  
→ FastPitch (mel spectrogram generation)  
→ HiFi-GAN (waveform generation)  
→ Audio Output

---

## 5. Core Components

### 5.1 Text Normalization
- Expand numbers (e.g., 123 → words)
- Handle names, slang, code-switching
- Language-specific rules

### 5.2 G2P Engine (MOST IMPORTANT)
- Custom phoneme inventory per language
- Handle:
  - Prenasalized consonants (mb, nd, ng)
  - Whistled fricatives (sv, zv)
  - Loanwords
- Override dictionary for accuracy

### 5.3 Acoustic Model (FastPitch)
- Converts phonemes → mel spectrogram
- Controllable pitch & duration

### 5.4 Vocoder (HiFi-GAN)
- Converts spectrogram → waveform
- Real-time capable

### 5.5 Alignment (MFA)
- Align text with audio
- Generate training-ready datasets

---

## 6. Data Requirements

### Minimum (Prototype)
- 2–5 hours per voice

### Production Quality
- 15–25 hours per voice (clean studio audio)

### Recording Specs
- 44.1kHz / 16-bit WAV
- Noise-free environment
- Consistent speaker tone

---

## 7. Training Strategy

### Phase 1 – Bootstrap
- Use small dataset
- Validate pronunciation + pipeline

### Phase 2 – Scale Voice
- Increase dataset size
- Improve prosody and fluency

### Phase 3 – Multi-language
- Repeat per language
- Reuse pipeline

---

## 8. Features

### MVP
- Single voice per language
- API-based synthesis
- Basic frontend normalization

### V1
- Multiple voices
- Emotion control (pitch, speed)
- Batch synthesis

### V2
- Code-switching support
- Accent control
- Voice cloning (internal only)

---

## 9. API Design

### Endpoint
POST /synthesize

### Input
{
  "text": "...",
  "language": "shona",
  "voice": "female_1",
  "speed": 1.0
}

### Output
- WAV / MP3 audio

---

## 10. Evaluation Metrics

- Pronunciation accuracy (native speaker validation)
- Naturalness (MOS score)
- Latency (<500ms target)
- Stability (no audio artifacts)

---

## 11. Risks

### Technical
- Poor G2P = bad output
- Low-quality data = unusable voices

### Mitigation
- Invest heavily in frontend
- Record high-quality datasets

---

## 12. Roadmap (90 Days)

### Month 1
- Build G2P + normalization
- Collect dataset
- Setup NeMo training

### Month 2
- Train first voice
- Evaluate + fix pronunciation

### Month 3
- Optimize quality
- Deploy API
- Prepare second language

---

## 13. Success Criteria

- Native speakers confirm authenticity
- Real-time inference achieved
- First commercial-ready voice deployed

---

## 14. Key Insight

**Your competitive advantage is NOT the model.**  
It is:
- Language understanding
- Phoneme accuracy
- Cultural correctness

---

## END
