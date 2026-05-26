"""
Convert existing audio files to the correct format for Coqui TTS training.
Takes any audio files in a directory, converts to 22050Hz mono 16-bit WAV,
and generates metadata if a transcriptions.csv is provided.

Usage:
    python scripts/prepare_audio.py --input_dir my_recordings/ --output_dir voice_dataset/wavs/ [--transcriptions transcriptions.csv]

transcriptions.csv format:
    filename,transcription
    recording1.wav,This is what I said in recording1.
"""

import os
import csv
import argparse
import soundfile as sf
import librosa
import numpy as np

TARGET_SR = 22050


def resample_file(input_path, output_path, target_sr=TARGET_SR):
    audio, sr = librosa.load(input_path, sr=None, mono=True)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    sf.write(output_path, audio, target_sr, subtype="PCM_16")
    return len(audio) / target_sr


def main():
    parser = argparse.ArgumentParser(description="Prepare audio for Coqui TTS training")
    parser.add_argument("--input_dir", required=True, help="Directory with source audio files")
    parser.add_argument("--output_dir", default="voice_dataset/wavs", help="Output directory for processed WAVs")
    parser.add_argument("--transcriptions", default=None, help="CSV with filename,transcription pairs")
    parser.add_argument("--metadata_out", default="voice_dataset/metadata_train.csv", help="Output metadata file path")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.metadata_out), exist_ok=True)

    transcriptions = {}
    if args.transcriptions and os.path.exists(args.transcriptions):
        with open(args.transcriptions, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    transcriptions[row[0].strip()] = row[1].strip()

    audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus"}
    files = sorted([
        f for f in os.listdir(args.input_dir)
        if os.path.splitext(f)[1].lower() in audio_extensions
    ])

    if not files:
        print(f"No audio files found in {args.input_dir}")
        return

    print(f"Found {len(files)} audio files. Converting to {TARGET_SR}Hz mono WAV...")

    with open(args.metadata_out, "w", encoding="utf-8", newline="") as meta_f:
        writer = csv.writer(meta_f, delimiter="|")
        writer.writerow(["file_name", "transcription", "speaker"])

        for i, fname in enumerate(files):
            in_path = os.path.join(args.input_dir, fname)
            out_name = f"sample_{i:04d}.wav"
            out_path = os.path.join(args.output_dir, out_name)

            duration = resample_file(in_path, out_path)
            text = transcriptions.get(fname, "")
            writer.writerow([out_name, text, "speaker"])
            print(f"  [{i+1}/{len(files)}] {out_name} ({duration:.1f}s)")

    print(f"\nDone! {len(files)} files prepared.")
    print(f"WAVs: {args.output_dir}/")
    print(f"Metadata: {args.metadata_out}")


if __name__ == "__main__":
    main()
