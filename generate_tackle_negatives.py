#!/usr/bin/env python3
"""
Generate hard-negative "tackle" confusable samples for wake word training.

These are phrases containing "tackle" WITHOUT the "listen up" prefix,
designed to teach the model that "tackle" alone (or with other prefixes)
should NOT trigger detection.

Generates all combinations of:
  - 7 confusable phrases
  - 9 OpenAI TTS voices
  - 5 speed variants
  = 315 WAV files in data/confusable/

Usage:
    python generate_tackle_negatives.py
    python generate_tackle_negatives.py --openai-key YOUR_KEY
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Set ffmpeg path before importing pydub
FFMPEG_DIR = r"C:\ffmpeg\bin"
if os.path.isdir(FFMPEG_DIR):
    os.environ["PATH"] = FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package required. Install with: pip install openai")
    sys.exit(1)

try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False

DATA_DIR = Path(__file__).parent / "data"

CONFUSABLE_PHRASES = [
    "tackle",
    "hey tackle",
    "up tackle",
    "go tackle",
    "the tackle",
    "that tackle",
    "oh tackle",
]

VOICES = ["alloy", "ash", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer"]
SPEEDS = [0.90, 0.95, 1.00, 1.05, 1.10]


def convert_mp3_to_wav(mp3_path, wav_path):
    """Convert MP3 to WAV (mono 16kHz)."""
    if HAS_PYDUB:
        audio = AudioSegment.from_mp3(str(mp3_path))
        audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
        audio.export(str(wav_path), format="wav")
        mp3_path.unlink()
        return True
    else:
        # Fallback: try soundfile
        try:
            import soundfile as sf
            import numpy as np
            # Can't read mp3 with soundfile, need pydub/ffmpeg
            print("  ERROR: pydub required for MP3->WAV conversion")
            return False
        except Exception as e:
            print(f"  Conversion error: {e}")
            return False


def generate_sample(client, text, output_path, voice="alloy", speed=1.0):
    """Generate one TTS WAV file."""
    try:
        response = client.audio.speech.create(
            model="tts-1-hd",
            voice=voice,
            input=text,
            speed=speed
        )
        mp3_path = output_path.with_suffix('.mp3')
        response.stream_to_file(str(mp3_path))

        if not convert_mp3_to_wav(mp3_path, output_path):
            print(f"  Warning: conversion failed for {output_path.name}")
            return False
        return True
    except Exception as e:
        print(f"  Error generating '{text}' ({voice}, {speed}): {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate tackle confusable negatives")
    parser.add_argument("--openai-key", help="OpenAI API key (or reads from user_config.json)")
    args = parser.parse_args()

    # Get API key: CLI arg > env var > user_config.json
    openai_key = args.openai_key or os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        config_path = Path(__file__).parent / "user_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            openai_key = config.get("openai_api_key", "")

    if not openai_key:
        print("ERROR: No OpenAI API key found.")
        print("  Use --openai-key, set OPENAI_API_KEY, or add to user_config.json")
        sys.exit(1)

    client = OpenAI(api_key=openai_key)
    output_dir = DATA_DIR / "confusable"
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(CONFUSABLE_PHRASES) * len(VOICES) * len(SPEEDS)
    print("=" * 60)
    print("Tackle Confusable Negative Generator")
    print("=" * 60)
    print(f"Phrases: {len(CONFUSABLE_PHRASES)}")
    print(f"Voices: {len(VOICES)}")
    print(f"Speeds: {len(SPEEDS)}")
    print(f"Total samples: {total}")
    print(f"Output: {output_dir}")
    print()

    generated = 0
    skipped = 0

    for phrase in CONFUSABLE_PHRASES:
        safe_phrase = phrase.replace(" ", "_")
        for voice in VOICES:
            for speed in SPEEDS:
                speed_str = f"{speed:.2f}"
                filename = f"confusable_{safe_phrase}_{voice}_{speed_str}.wav"
                output_path = output_dir / filename

                if output_path.exists():
                    skipped += 1
                    continue

                idx = generated + skipped + 1
                print(f"  [{idx}/{total}] '{phrase}' ({voice}, {speed_str})")
                if generate_sample(client, phrase, output_path, voice, speed):
                    generated += 1
                else:
                    print(f"    FAILED: {filename}")

    print(f"\n{'=' * 60}")
    print(f"Done! Generated: {generated}, Skipped (existing): {skipped}")
    print(f"Total confusable files: {generated + skipped}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
