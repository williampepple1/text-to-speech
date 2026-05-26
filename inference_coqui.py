"""
Inference script: Use your fine-tuned XTTS v2 voice clone.

Place your fine-tuned model files (best_model.pth, config.json, etc.)
in the xtts_finetuned/ directory, then run:

    python inference_coqui.py --text "Hello world" --output output.wav

Or use the VoiceCloner class in your own code.
"""

import os
import sys
import argparse

SAMPLE_RATE = 22050


class VoiceCloner:
    """
    Loads a fine-tuned XTTS v2 model and generates speech.
    Also works with the base XTTS v2 model for zero-shot cloning.

    Usage:
        cloner = VoiceCloner(model_dir="xtts_finetuned")
        cloner.speak("Hello, this is my voice.", output_path="output.wav")
    """

    def __init__(self, model_dir: str = "xtts_finetuned", use_base: bool = False):
        from TTS.api import TTS

        self.model_dir = model_dir
        self.use_base = use_base

        if use_base:
            self.model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
            self.tts = TTS(model_name=self.model_name, progress_bar=False)
        else:
            model_path = os.path.join(model_dir, "best_model.pth")
            config_path = os.path.join(model_dir, "config.json")

            if not os.path.exists(model_path):
                candidates = []
                for root, _, files in os.walk(model_dir):
                    for f in files:
                        if f.endswith(".pth"):
                            candidates.append(os.path.join(root, f))
                if candidates:
                    model_path = sorted(candidates)[-1]
                else:
                    raise FileNotFoundError(f"No .pth model found in {model_dir}")

            if not os.path.exists(config_path):
                json_candidates = []
                for root, _, files in os.walk(model_dir):
                    for f in files:
                        if f == "config.json":
                            json_candidates.append(os.path.join(root, f))
                config_path = json_candidates[0] if json_candidates else None

            print(f"Loading model from: {model_path}")
            if config_path:
                print(f"Config from: {config_path}")

            self.tts = TTS(
                model_path=model_path,
                config_path=config_path,
                progress_bar=False,
            )

    def speak(self, text: str, output_path: str = "output.wav",
              speaker_wav: str = None, language: str = "en"):
        """
        Generate speech from text.

        Args:
            text: The text to synthesize
            output_path: Where to save the WAV file
            speaker_wav: Reference speaker audio for voice cloning.
                         If None, uses the fine-tuned voice directly.
            language: Language code (en, es, fr, de, it, pt, pl, tr, ru, nl,
                      cs, ar, zh-cn, hu, ko, ja, hi)
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        if speaker_wav and os.path.exists(speaker_wav):
            self.tts.tts_to_file(
                text=text,
                speaker_wav=speaker_wav,
                language=language,
                file_path=output_path,
            )
        else:
            self.tts.tts_to_file(text=text, file_path=output_path)

        print(f"Saved: {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(description="Coqui XTTS v2 Voice Clone Inference")
    parser.add_argument("--text", "-t", type=str, required=True,
                        help="Text to synthesize")
    parser.add_argument("--output", "-o", type=str, default="output.wav",
                        help="Output WAV file path")
    parser.add_argument("--model_dir", "-m", type=str, default="xtts_finetuned",
                        help="Directory containing fine-tuned model")
    parser.add_argument("--speaker_wav", "-s", type=str, default=None,
                        help="Reference WAV for voice cloning style")
    parser.add_argument("--language", "-l", type=str, default="en",
                        help="Language code")
    parser.add_argument("--base_model", action="store_true",
                        help="Use base XTTS v2 instead of fine-tuned model")

    args = parser.parse_args()

    cloner = VoiceCloner(model_dir=args.model_dir, use_base=args.base_model)
    cloner.speak(
        text=args.text,
        output_path=args.output,
        speaker_wav=args.speaker_wav,
        language=args.language,
    )


if __name__ == "__main__":
    main()
