# 06: Speech & Audio Deep Learning

End-to-end deep learning for audio — from raw waveforms to text-to-speech synthesis, automatic speech recognition, and voice cloning.

## Notebooks

| Folder | Topic | Key Models |
|--------|-------|-----------|
| `01-Basic-Spectogram/` | Audio Feature Extraction | Mel spectrogram, MFCC, log-Mel |
| `02-TTS/` | Text-to-Speech | WaveNet, Tacotron 2, TSync2 |
| `03-STT/` | Speech-to-Text | Whisper (from scratch + OpenAI API) |
| `04-VoiceCloning/` | Voice Cloning | FreeVC, OpenVoice |

## Audio Processing Pipeline

```
Raw Audio (waveform)
    │
    ▼  Short-Time Fourier Transform (STFT)
Spectrogram  (time × frequency)
    │
    ▼  Mel filterbank (80 or 128 filters)
Log-Mel Spectrogram    ← most models use this as input
    │
    ▼  [optional] Discrete Cosine Transform
MFCC (Mel-Frequency Cepstral Coefficients)
```

## Model Architectures

### Text-to-Speech

```
Text → Phonemes → Encoder → Attention → Decoder → Mel Spectrogram → Vocoder → Audio
                                                                       ↑
                                                              WaveNet / WaveRNN
```

| Model | Year | Innovation |
|-------|------|-----------|
| WaveNet | 2016 | Autoregressive dilated causal convolution; first neural vocoder |
| Tacotron 2 | 2018 | Seq2seq + location-sensitive attention; LSTM encoder-decoder |
| FastSpeech 2 | 2021 | Non-autoregressive; duration/pitch/energy predictors |

### Speech-to-Text (Whisper)

```
Audio → Log-Mel Spectrogram → Transformer Encoder → Transformer Decoder → Text
                                   (Whisper)               (with BPE vocab)
```

- **Weak supervision at scale**: trained on 680k hours of multilingual audio from the web
- **Multitask**: transcription, translation, language ID, timestamp prediction

### Voice Cloning

```
Reference Audio → Speaker Encoder → Speaker Embedding
                                          ↓
Text ──────────────────────────────→ TTS Model → Cloned Voice
```

| Model | Approach |
|-------|---------|
| FreeVC | Information-bottleneck voice conversion; disentangle content from speaker |
| OpenVoice | Tone color embedding; flexible style transfer |

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Mel spectrogram** | Log-frequency spectrogram matching human auditory perception |
| **Vocoder** | Neural model to convert Mel spectrogram back to waveform |
| **Attention (TTS)** | Aligns encoder (text) positions to decoder (audio) positions |
| **CTC loss** | Connectionist Temporal Classification for sequence alignment without forced alignment |
| **BPE** | Byte-Pair Encoding tokenizer used by Whisper for multilingual text |

## References

| Paper | Link |
|-------|------|
| WaveNet — A Generative Model for Raw Audio | [arxiv](https://arxiv.org/abs/1609.03499) |
| Tacotron 2 — Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram | [arxiv](https://arxiv.org/abs/1712.05884) |
| Whisper — Robust Speech Recognition via Large-Scale Weak Supervision | [arxiv](https://arxiv.org/abs/2212.04356) |
| FreeVC — Towards High-Quality Text-Free One-Shot Voice Conversion | [arxiv](https://arxiv.org/abs/2210.15418) |
