"""
Kaggle Notebook: Fine-tune Coqui TTS XTTS v2 on your voice.

This script is designed to run on Kaggle with GPU (T4 x2 or P100).
It fine-tunes the XTTS v2 model on your voice dataset.

=== HOW TO USE ON KAGGLE ===

1. Upload your voice_dataset/ folder as a Kaggle Dataset
   - Go to kaggle.com -> Your Profile -> Datasets -> New Dataset
   - Upload the voice_dataset/ folder
   - Note the dataset path (e.g., "yourname/voice-dataset")

2. Create a new Kaggle Notebook:
   - GPU: T4 x2 (recommended) or P100
   - Add your dataset as input

3. Run this script cell by cell or as a single block.

=== EXPECTED OUTPUT ===
  - Fine-tuned model in ./xtts_finetuned/
  - Download these files for local inference.

=== MINIMUM REQUIREMENTS ===
  - At least 6 minutes of clean speech audio (more is better)
  - 16GB+ GPU VRAM (T4 x2 works, P100 works)
  - Dataset formatted as voice_dataset/wavs/*.wav + metadata_train.csv
"""

import os
import shutil
import subprocess
import sys

# ============================================================
# CELL 1: Install dependencies
# ============================================================

def install_dependencies():
    print("=" * 60)
    print("CELL 1: Installing dependencies")
    print("=" * 60)

    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "TTS>=0.22.0",
        "soundfile>=0.12.0",
        "librosa>=0.10.0",
        "scipy",
        "numpy",
        "pydub",
        "einops",
        "transformers>=4.30.0",
        "tokenizers>=0.13.0",
        "sentencepiece",
        "trainer>=0.0.32",
        "coqpit>=0.0.15",
    ])

    import torch
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")


# ============================================================
# CELL 2: Clone Coqui TTS repo and set up
# ============================================================

def setup_tts_repo():
    print("=" * 60)
    print("CELL 2: Setting up Coqui TTS repository")
    print("=" * 60)

    REPO_DIR = "/kaggle/working/TTS"

    if not os.path.exists(REPO_DIR):
        subprocess.check_call([
            "git", "clone", "https://github.com/coqui-ai/TTS.git", REPO_DIR
        ])
    else:
        print("Repo already cloned.")

    os.chdir(REPO_DIR)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])

    return REPO_DIR


# ============================================================
# CELL 3: Verify and copy dataset
# ============================================================

def prepare_dataset():
    """
    Expects voice_dataset/ at /kaggle/input/voice-dataset/voice_dataset/
    with structure:
        voice_dataset/
            wavs/
                sample_0001.wav
                ...
            metadata_train.csv  (| separated: file_name|transcription|speaker_name)
    """
    print("=" * 60)
    print("CELL 3: Preparing dataset")
    print("=" * 60)

    import glob
    import csv

    DEST = "/kaggle/working/voice_dataset"
    SOURCE_CANDIDATES = [
        "/kaggle/input/voice-dataset/voice_dataset",
        "/kaggle/input/voice_dataset",
        "/kaggle/input/voice-dataset",
        "/kaggle/working/voice_dataset",
    ]

    source = None
    for path in SOURCE_CANDIDATES:
        if os.path.isdir(path) and os.path.isdir(os.path.join(path, "wavs")):
            source = path
            break

    if source is None:
        print("ERROR: Could not find voice dataset!")
        print("Looked in:", SOURCE_CANDIDATES)
        print("\nMake sure your dataset is structured as:")
        print("  voice_dataset/")
        print("    wavs/")
        print("      sample_0001.wav")
        print("      ...")
        print("    metadata_train.csv")
        return None

    if source != DEST:
        if os.path.exists(DEST):
            shutil.rmtree(DEST)
        shutil.copytree(source, DEST)

    wavs_dir = os.path.join(DEST, "wavs")
    metadata_file = os.path.join(DEST, "metadata_train.csv")

    if not os.path.exists(metadata_file):
        print("ERROR: metadata_train.csv not found!")
        print(f"Expected at: {metadata_file}")
        return None

    wav_files = glob.glob(os.path.join(wavs_dir, "*.wav"))
    print(f"Dataset source: {source}")
    print(f"WAV files found: {len(wav_files)}")

    total_duration = 0
    for wf in wav_files:
        import soundfile as sf
        info = sf.info(wf)
        total_duration += info.duration
    print(f"Total audio duration: {total_duration / 60:.1f} minutes")

    with open(metadata_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        lines = list(reader)
        print(f"Metadata entries: {len(lines) - 1}")  # -1 for header

    if total_duration < 60:
        print(f"\nWARNING: Only {total_duration:.1f} seconds of audio.")
        print("XTTS v2 works best with 6+ minutes. Quality may be lower.")
    elif total_duration < 360:
        print(f"\nNote: {total_duration / 60:.1f} minutes is a good start.")
        print("For best results, aim for 10+ minutes.")
    else:
        print(f"\nGreat! {total_duration / 60:.1f} minutes is excellent for fine-tuning.")

    return DEST


# ============================================================
# CELL 4: Download XTTS v2 base model and configure
# ============================================================

def configure_training(dataset_path):
    print("=" * 60)
    print("CELL 4: Downloading XTTS v2 and configuring training")
    print("=" * 60)

    from TTS.api import TTS
    import torch

    MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

    print(f"Downloading base model: {MODEL_NAME}")
    tts = TTS(model_name=MODEL_NAME, progress_bar=True)

    model_dir = tts.synthesizer.tts_model_dir if hasattr(tts.synthesizer, "tts_model_dir") else None
    if model_dir:
        print(f"Base model downloaded to: {model_dir}")

    CONFIG_PATH = "/kaggle/working/xtts_finetune_config.json"
    config = {
        "model": "xtts",
        "run_name": "xtts_v2_finetuned",
        "output_path": "/kaggle/working/xtts_finetuned",
        "datasets": [
            {
                "name": "voice_clone",
                "path": dataset_path,
                "meta_file_train": "metadata_train.csv",
                "meta_file_val": None,
                "language": "en",
            }
        ],
        "audio": {
            "sample_rate": 22050,
            "fft_size": 1024,
            "win_length": 1024,
            "hop_length": 256,
            "num_mels": 80,
        },
        "trainer": {
            "epochs": 50,
            "batch_size": 4,
            "eval_batch_size": 4,
            "mixed_precision": True,
            "save_step": 500,
            "print_step": 50,
            "plot_step": 500,
            "checkpoint": True,
            "lr": 5e-6,
            "lr_scheduler": "cosine",
            "warmup_steps": 100,
            "weight_decay": 1e-6,
            "grad_clip": 1.0,
        },
        "gpt": {
            "num_heads": 8,
            "num_layers": 30,
            "hidden_dim": 1024,
            "use_perceiver": False,
            "max_text_tokens": 400,
            "max_audio_tokens": 500,
            "gpt_max_audio_tokens": 600,
            "gpt_max_text_tokens": 400,
            "gpt_max_prompt_tokens": 150,
            "num_chars": 256,
        },
    }

    import json
    os.makedirs("/kaggle/working/xtts_finetuned", exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Config saved to: {CONFIG_PATH}")
    return CONFIG_PATH, tts


# ============================================================
# CELL 5: Run fine-tuning
# ============================================================

def run_finetuning(config_path, tts_model):
    print("=" * 60)
    print("CELL 5: Running XTTS v2 fine-tuning")
    print("=" * 60)

    import torch
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    from TTS.tts.datasets import load_tts_samples
    from TTS.tts.utils.text.tokenizer import VoiceBpeTokenizer
    from TTS.trainer import Trainer, TrainingArgs
    import json

    with open(config_path, "r") as f:
        config_dict = json.load(f)

    dataset_path = config_dict["datasets"][0]["path"]

    config = XttsConfig()
    config.output_path = config_dict["output_path"]
    config.run_name = config_dict["run_name"]
    config.audio.sample_rate = config_dict["audio"]["sample_rate"]

    for k, v in config_dict["trainer"].items():
        if hasattr(config, k):
            setattr(config, k, v)

    config.datasets = config_dict["datasets"]

    tokenizer = VoiceBpeTokenizer()

    train_samples, eval_samples = load_tts_samples(
        config.datasets,
        eval_split=True,
        eval_split_max_size=config_dict["trainer"]["eval_batch_size"],
        eval_split_size=0.01,
    )

    print(f"Training samples: {len(train_samples)}")
    print(f"Eval samples: {len(eval_samples)}")

    model = Xtts.init_from_config(config)
    model.init_multispeaker(config)

    trainer = Trainer(
        args=TrainingArgs(
            output_path=config.output_path,
            run_name=config.run_name,
            epochs=config_dict["trainer"]["epochs"],
            batch_size=config_dict["trainer"]["batch_size"],
            eval_batch_size=config_dict["trainer"]["eval_batch_size"],
            mixed_precision=config_dict["trainer"]["mixed_precision"],
            save_step=config_dict["trainer"]["save_step"],
            print_step=config_dict["trainer"]["print_step"],
            plot_step=config_dict["trainer"]["plot_step"],
            lr=config_dict["trainer"]["lr"],
            lr_scheduler=config_dict["trainer"]["lr_scheduler"],
            warmup_steps=config_dict["trainer"]["warmup_steps"],
            weight_decay=config_dict["trainer"]["weight_decay"],
            grad_clip=config_dict["trainer"]["grad_clip"],
            checkpoint=config_dict["trainer"]["checkpoint"],
        ),
        config=config,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        training_assets={"tokenizer": tokenizer},
    )

    print("\nStarting training...")
    trainer.fit()

    print("\nTraining complete!")
    print(f"Model saved to: {config.output_path}")


# ============================================================
# CELL 6: Test the fine-tuned model
# ============================================================

def test_model():
    print("=" * 60)
    print("CELL 6: Testing fine-tuned model")
    print("=" * 60)

    from TTS.api import TTS
    import torch

    MODEL_PATH = "/kaggle/working/xtts_finetuned/best_model.pth"
    CONFIG_PATH = "/kaggle/working/xtts_finetuned/config.json"

    if not os.path.exists(MODEL_PATH):
        print(f"Looking for model...")
        import glob
        candidates = glob.glob("/kaggle/working/xtts_finetuned/**/*.pth", recursive=True)
        if candidates:
            MODEL_PATH = candidates[0]
            print(f"Found: {MODEL_PATH}")
        else:
            print("No model found! Training may have failed.")
            return

    test_text = "Hello, this is my cloned voice. I trained this model using Coqui TTS on Kaggle."
    output_path = "/kaggle/working/test_output.wav"

    sample_wav = None
    import glob
    wavs = glob.glob("/kaggle/working/voice_dataset/wavs/*.wav")
    if wavs:
        sample_wav = wavs[0]
        print(f"Using reference speaker audio: {sample_wav}")

    try:
        tts = TTS(
            model_path=MODEL_PATH,
            config_path=CONFIG_PATH,
            progress_bar=True,
        )

        if sample_wav:
            tts.tts_to_file(
                text=test_text,
                speaker_wav=sample_wav,
                language="en",
                file_path=output_path,
            )
        else:
            tts.tts_to_file(text=test_text, file_path=output_path)

        print(f"\nTest audio saved to: {output_path}")
        print("You can play this in Kaggle to check quality.")

        from IPython.display import Audio
        return Audio(output_path)

    except Exception as e:
        print(f"Error during inference: {e}")
        print("\nTrying alternative approach with base model...")
        try:
            tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
            tts.tts_to_file(text=test_text, file_path=output_path)
            print(f"Test audio (base model) saved to: {output_path}")
        except Exception as e2:
            print(f"Alternative also failed: {e2}")


# ============================================================
# CELL 7: Package model for download
# ============================================================

def package_model():
    print("=" * 60)
    print("CELL 7: Packaging model for download")
    print("=" * 60)

    import glob
    import json

    PACKAGE_DIR = "/kaggle/working/xtts_finetuned_export"
    os.makedirs(PACKAGE_DIR, exist_ok=True)

    model_files = (
        glob.glob("/kaggle/working/xtts_finetuned/best_model*.pth") +
        glob.glob("/kaggle/working/xtts_finetuned/checkpoint_*.pth") +
        glob.glob("/kaggle/working/xtts_finetuned/config.json") +
        glob.glob("/kaggle/working/xtts_finetuned/speakers.pth") +
        glob.glob("/kaggle/working/xtts_finetuned/vocab.json") +
        glob.glob("/kaggle/working/xtts_finetuned/*.json")
    )

    copied = 0
    for f in set(model_files):
        if os.path.isfile(f):
            dest = os.path.join(PACKAGE_DIR, os.path.basename(f))
            shutil.copy2(f, dest)
            copied += 1
            print(f"  Copied: {os.path.basename(f)}")

    if copied == 0:
        print("No model files found in xtts_finetuned/")
        print("Searching recursively...")
        all_pth = glob.glob("/kaggle/working/xtts_finetuned/**/*.pth", recursive=True)
        all_json = glob.glob("/kaggle/working/xtts_finetuned/**/*.json", recursive=True)
        for f in all_pth + all_json:
            dest = os.path.join(PACKAGE_DIR, os.path.basename(f))
            shutil.copy2(f, dest)
            copied += 1
            print(f"  Copied: {os.path.basename(f)}")

    info = {
        "base_model": "tts_models/multilingual/multi-dataset/xtts_v2",
        "description": "Fine-tuned XTTS v2 voice clone",
        "sample_rate": 22050,
        "files": os.listdir(PACKAGE_DIR),
    }
    with open(os.path.join(PACKAGE_DIR, "model_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    zip_path = "/kaggle/working/xtts_finetuned_export.zip"
    shutil.make_archive(
        zip_path.replace(".zip", ""), "zip", PACKAGE_DIR
    )
    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"\nExported to: {zip_path} ({zip_size_mb:.1f} MB)")
    print("\nDownload this zip file from the Kaggle output tab!")


# ============================================================
# MAIN: Run all cells
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune XTTS v2 on Kaggle")
    parser.add_argument("--step", type=str, default="all",
                        choices=["all", "install", "setup", "dataset", "configure",
                                 "train", "test", "package"],
                        help="Which step to run")
    args = parser.parse_args()

    steps = {
        "install": install_dependencies,
        "setup": setup_tts_repo,
        "dataset": prepare_dataset,
        "configure": lambda: configure_training(prepare_dataset()),
        "train": lambda: run_finetuning(*configure_training(prepare_dataset())),
        "test": test_model,
        "package": package_model,
    }

    if args.step == "all":
        install_dependencies()
        setup_tts_repo()
        dataset_path = prepare_dataset()
        if dataset_path is None:
            print("ABORT: Dataset not found. Fix and retry.")
            sys.exit(1)
        config_path, tts_model = configure_training(dataset_path)
        run_finetuning(config_path, tts_model)
        test_model()
        package_model()
    elif args.step in steps:
        steps[args.step]()
