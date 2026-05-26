import sounddevice as sd
import soundfile as sf
import numpy as np
import os
import csv
from datetime import datetime

SAMPLE_RATE = 22050
DATASET_DIR = "voice_dataset"
WAVS_DIR = os.path.join(DATASET_DIR, "wavs")
METADATA_FILE = os.path.join(DATASET_DIR, "metadata_train.csv")

SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "I enjoy reading books about science and technology.",
    "Hello, my name is and I am recording my voice for a project.",
    "The weather today is pleasant with a gentle breeze.",
    "Music can evoke strong emotions and memories.",
    "Learning a new programming language takes time and practice.",
    "Coffee is one of the most popular beverages in the world.",
    "The ocean waves crashed against the rocky shore.",
    "Artificial intelligence is transforming many industries.",
    "Reading before bed helps me fall asleep faster.",
    "A journey of a thousand miles begins with a single step.",
    "The library was quiet except for the turning of pages.",
    "Cooking a good meal requires fresh ingredients and patience.",
    "The mountains were covered in a blanket of fresh snow.",
    "She opened the window to let in the morning light.",
    "Dogs are known for their loyalty and companionship.",
    "The internet has changed the way we communicate forever.",
    "Raindrops tapped gently against the window pane.",
    "Birds sang their morning chorus as the sun rose.",
    "Success comes from hard work and dedication.",
    "The garden bloomed with vibrant flowers in spring.",
    "I love the smell of freshly baked bread in the morning.",
    "Technology evolves faster than we can keep up with.",
    "The stars shone brightly in the clear night sky.",
    "Exercise is important for both physical and mental health.",
    "A good laugh can brighten even the darkest day.",
    "The river flowed gently through the green valley.",
    "Books open doors to new worlds and perspectives.",
    "Time spent with family is never wasted.",
    "The city skyline looked beautiful at sunset.",
]


def ensure_dirs():
    os.makedirs(WAVS_DIR, exist_ok=True)


def load_progress():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="|")
            return {row[0] for row in reader}
    return set()


def save_metadata(filename, text):
    file_exists = os.path.exists(METADATA_FILE)
    with open(METADATA_FILE, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="|")
        if not file_exists:
            writer.writerow(["file_name", "transcription", "speaker"])
        writer.writerow([filename, text, "speaker"])


def record_sample(duration_sec=10, sample_rate=SAMPLE_RATE):
    print(f"\nRecording for up to {duration_sec} seconds... Press Ctrl+C to stop early.")
    recording = sd.rec(
        int(duration_sec * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    return recording.squeeze()


def main():
    ensure_dirs()
    completed = load_progress()
    remaining = [(i, s) for i, s in enumerate(SENTENCES) if f"sample_{i:04d}.wav" not in completed]

    if not remaining:
        print("All samples already recorded!")
        print(f"Dataset saved in '{DATASET_DIR}/'")
        print(f"Total samples: {len(SENTENCES)}")
        return

    print(f"=== Voice Dataset Recorder ===")
    print(f"Completed: {len(SENTENCES) - len(remaining)}/{len(SENTENCES)}")
    print(f"Remaining: {len(remaining)}")
    print(f"Output dir: {DATASET_DIR}/")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print("==============================\n")

    for idx, sentence in remaining:
        filename = f"sample_{idx:04d}.wav"
        filepath = os.path.join(WAVS_DIR, filename)

        print(f"\n--- Sample {idx + 1}/{len(SENTENCES)} ---")
        print(f"Text: {sentence}")
        input("Press Enter when ready to record...")

        audio = record_sample()

        trimmed = np.trim_zeros(audio, trim="fb")
        if len(trimmed) == 0:
            print("No audio detected, skipping.")
            continue

        sf.write(filepath, trimmed, SAMPLE_RATE)
        save_metadata(filename, sentence)
        print(f"Saved: {filename} ({len(trimmed) / SAMPLE_RATE:.1f}s)")

    print(f"\nDone! Dataset saved in '{DATASET_DIR}/'")
    print(f"Total samples: {len(SENTENCES)}")
    print("\nTo train on Kaggle:")
    print("  1. Zip the voice_dataset folder")
    print("  2. Upload as a Kaggle dataset or include in notebook")
    print("  3. Run kaggle/train_xtts.py")


if __name__ == "__main__":
    main()
