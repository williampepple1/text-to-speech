# Text to Speech Voice Clone

Fine-tune [Coqui TTS XTTS v2](https://github.com/coqui-ai/TTS) on your own voice — trained on Kaggle, run locally.

## How it works

1. **Record** your voice locally (30 sentences, ~6 minutes)
2. **Train** the model on Kaggle's free GPU (T4 x2)
3. **Download** the fine-tuned model and generate speech locally

The FastAPI server (`main.py`) auto-detects the fine-tuned model and uses it instead of the built-in pyttsx3 engine.

---

## Prerequisites

- Python 3.9+
- A [Kaggle](https://kaggle.com) account (free GPU hours)
- A microphone

---

## Step 1: Record your voice

Install the recording tools:

```bash
pip install -r requirements-scripts.txt
```

Record yourself reading 30 sentences:

```bash
python scripts/record_voice.py
```

- Speak naturally, at a consistent pace and volume
- Record in a quiet room with minimal background noise
- Each sentence gets saved as a separate 22050Hz mono WAV
- You can stop and resume — progress is saved automatically

Output:
```
voice_dataset/
├── wavs/
│   ├── sample_0000.wav
│   ├── sample_0001.wav
│   └── ...
└── metadata_train.csv
```

> **Tip:** You can also bring your own audio files. Use `scripts/prepare_audio.py` to convert them to the right format.

---

## Step 2: Train on Kaggle

### 2a. Upload your dataset

1. Go to [kaggle.com](https://kaggle.com) → **Your Profile** → **Datasets** → **New Dataset**
2. Upload the `voice_dataset/` folder
3. Note the dataset path (e.g. `yourusername/voice-dataset`)

### 2b. Run the notebook

1. Go to **Kaggle** → **Create** → **New Notebook**
2. Set **Accelerator** to `T4 x2` (right panel)
3. Add your dataset as input (right panel → **Input** → **Add Dataset**)
4. Upload `kaggle/train_xtts.ipynb` or copy its cells into your notebook
5. Run all cells in order

The notebook will:
- Install PyTorch and Coqui TTS
- Download the XTTS v2 base model (~1.9GB)
- Fine-tune on your voice (50 epochs, ~30-60 minutes on T4)
- Test the model and package it into a zip

### 2c. Download the model

In the Kaggle output tab, download `xtts_finetuned_export.zip`. Extract it into your local project root so you have:

```
xtts_finetuned/
├── best_model.pth
├── config.json
└── ...
```

---

## Step 3: Use your cloned voice

### Command line

```bash
pip install -r requirements-coqui.txt
python inference_coqui.py -t "Hello, this is my cloned voice." -o output.wav
```

Options:
```
-t, --text        Text to synthesize (required)
-o, --output      Output WAV path (default: output.wav)
-m, --model_dir   Model directory (default: xtts_finetuned)
-l, --language    Language code (default: en)
-s, --speaker_wav Reference WAV for style (optional)
--base_model      Use base XTTS v2 instead of fine-tuned model
```

### Python API

```python
from inference_coqui import VoiceCloner

cloner = VoiceCloner(model_dir="xtts_finetuned")
cloner.speak("Hello world", output_path="hello.wav")

# With a reference clip for style matching
cloner.speak("Goodbye", output_path="bye.wav", speaker_wav="voice_dataset/wavs/sample_0000.wav")
```

### FastAPI server

The server auto-detects the model. Just drop `xtts_finetuned/` in the project root and start:

```bash
uvicorn main:app
```

- `POST /generate-audio/` — `{"text": "your text here"}`
- `GET /audio/{file_name}` — download generated audio

If no fine-tuned model is found, it falls back to pyttsx3 (system TTS voice).

---

## File reference

| File | Purpose |
|------|---------|
| `scripts/record_voice.py` | Record 30 sentences into training data |
| `scripts/prepare_audio.py` | Convert existing audio to 22050Hz mono WAV |
| `kaggle/train_xtts.ipynb` | Kaggle notebook for fine-tuning |
| `kaggle/train_xtts.py` | Same training logic as a standalone script |
| `inference_coqui.py` | Local inference with the fine-tuned model |
| `main.py` | FastAPI server (auto-detects Coqui model) |
| `requirements-coqui.txt` | Coqui TTS inference dependencies |
| `requirements-scripts.txt` | Recording/prep tool dependencies |
| `requirements.txt` | Core server dependencies |

---

## Troubleshooting

**"No module named TTS"** — Run `pip install -r requirements-coqui.txt`. On Windows, Coqui TTS needs at least Python 3.9 and PyTorch with CUDA or CPU support.

**Training fails with OOM on Kaggle** — Reduce `batch_size` in the config (try 2 or 1). Make sure you're using T4 x2, not a single T4.

**Voice quality is poor** — Aim for 10+ minutes of audio, clean recordings with no background noise, and accurate transcriptions in `metadata_train.csv`.

**On Windows, recording may fail** — Install portaudio: `pip install pipwin` then `pipwin install pyaudio`, or use `scripts/prepare_audio.py` with pre-recorded files.
