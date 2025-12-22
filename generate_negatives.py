#!/usr/bin/env python3
"""
Generate Negative Training Samples for Wake Word Detection

This script generates three types of negative samples:
1. Confusable words - Similar-sounding phrases (hard negatives)
2. Common phrases - Everyday speech that should NOT trigger
3. Background noise/silence - For robustness

Usage:
    python generate_negatives.py --wake-word "Hey Nevyx" --openai-key YOUR_KEY
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Try to import required packages
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from elevenlabs.client import ElevenLabs
    HAS_ELEVENLABS = True
except ImportError:
    HAS_ELEVENLABS = False

try:
    import torchaudio
    import torch
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False

# Import augmentation module for confusable word list
try:
    from augmentation import generate_confusable_words, get_common_phrases
except ImportError:
    # Fallback implementations
    def generate_confusable_words(wake_word):
        """Generate phonetically similar words."""
        confusables = []
        words = wake_word.lower().split()

        if words[0] in ['hey', 'hi', 'hay']:
            prefixes = ['hey', 'hi', 'hay', 'say', 'day', 'may', 'way', 'pay', 'okay', 'a']

            if len(words) > 1:
                target = words[1]
                similar_names = {
                    'nevyx': ['nexus', 'netflix', 'nervous', 'devin', 'kevin', 'devon', 'heaven', 'seven', 'levis'],
                    'jarvis': ['service', 'harvest', 'carvis', 'marvis', 'jarvas', 'travis', 'jarred'],
                }
                similar = similar_names.get(target, [])

                for prefix in prefixes:
                    if prefix != words[0]:
                        confusables.append(f"{prefix} {target}")
                    for sim in similar:
                        confusables.append(f"{prefix} {sim}")
                        confusables.append(f"{words[0]} {sim}")

        confusables.extend([
            "okay", "hey there", "hey you", "hey now", "no way", "anyway",
        ])

        return list(set([c for c in confusables if c.lower() != wake_word.lower()]))

    def get_common_phrases():
        """Get common spoken phrases for negative samples."""
        return [
            "hello", "hi there", "good morning", "how are you",
            "what time is it", "play some music", "turn on the lights",
            "I don't know", "sounds good", "thank you", "you're welcome",
        ]


DATA_DIR = Path(__file__).parent / "data"


def convert_mp3_to_wav(mp3_path, wav_path):
    """Convert MP3 to WAV using pydub (requires ffmpeg)."""
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_mp3(str(mp3_path))
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(str(wav_path), format="wav")
        mp3_path.unlink()
        return True
    except Exception as e:
        print(f"  Conversion error: {e}")
        return False


def generate_with_openai(client, text, output_path, voice="alloy", speed=1.0):
    """Generate TTS audio using OpenAI."""
    try:
        response = client.audio.speech.create(
            model="tts-1-hd",
            voice=voice,
            input=text,
            speed=speed
        )

        mp3_path = output_path.with_suffix('.mp3')
        response.stream_to_file(str(mp3_path))

        # Convert to WAV using pydub
        if not convert_mp3_to_wav(mp3_path, output_path):
            print(f"  Warning: keeping MP3 (conversion failed)")

        return True
    except Exception as e:
        print(f"  Error generating '{text}': {e}")
        return False


def generate_confusable_samples(wake_word, openai_key, output_dir):
    """Generate TTS samples for confusable words."""
    if not HAS_OPENAI:
        print("ERROR: openai package required. Install with: pip install openai")
        return 0

    client = OpenAI(api_key=openai_key)
    confusables = generate_confusable_words(wake_word)

    print(f"\nGenerating {len(confusables)} confusable word samples...")
    print(f"Output directory: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    voices = ["alloy", "echo", "fable", "nova", "onyx", "shimmer"]
    speeds = [0.9, 1.0, 1.1]

    for i, text in enumerate(confusables):
        # Use different voice/speed combinations for variety
        voice = voices[i % len(voices)]
        speed = speeds[i % len(speeds)]

        # Clean filename
        safe_name = "".join(c if c.isalnum() else "_" for c in text)[:30]
        output_path = output_dir / f"confusable_{safe_name}_{voice}.wav"

        if output_path.exists():
            print(f"  Skipping (exists): {text}")
            count += 1
            continue

        print(f"  [{i+1}/{len(confusables)}] Generating: '{text}' ({voice})")
        if generate_with_openai(client, text, output_path, voice, speed):
            count += 1

    return count


def generate_negative_samples(openai_key, output_dir):
    """Generate TTS samples for common phrases (negatives)."""
    if not HAS_OPENAI:
        print("ERROR: openai package required. Install with: pip install openai")
        return 0

    client = OpenAI(api_key=openai_key)
    phrases = get_common_phrases()

    print(f"\nGenerating {len(phrases)} common phrase samples (negatives)...")
    print(f"Output directory: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    voices = ["alloy", "echo", "fable", "nova", "onyx", "shimmer"]

    for i, text in enumerate(phrases):
        voice = voices[i % len(voices)]

        safe_name = "".join(c if c.isalnum() else "_" for c in text)[:30]
        output_path = output_dir / f"negative_{safe_name}_{voice}.wav"

        if output_path.exists():
            print(f"  Skipping (exists): {text}")
            count += 1
            continue

        print(f"  [{i+1}/{len(phrases)}] Generating: '{text}' ({voice})")
        if generate_with_openai(client, text, output_path, voice):
            count += 1

    return count


def generate_silence_samples(output_dir, count=10):
    """Generate silence/near-silence samples."""
    print(f"\nGenerating {count} silence samples...")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not HAS_TORCHAUDIO:
        print("  Warning: torchaudio not available, skipping silence generation")
        return 0

    generated = 0
    for i in range(count):
        output_path = output_dir / f"silence_{i:03d}.wav"
        if output_path.exists():
            continue

        # Generate 2 seconds of near-silence with slight noise
        samples = 16000 * 2  # 2 seconds at 16kHz
        noise_level = 0.001 * (i + 1) / count  # Varying noise levels
        audio = torch.randn(1, samples) * noise_level

        torchaudio.save(str(output_path), audio, 16000)
        generated += 1

    print(f"  Generated {generated} silence samples")
    return generated


def main():
    parser = argparse.ArgumentParser(description="Generate negative training samples")
    parser.add_argument("--wake-word", default="Hey Nevyx", help="Wake word to generate confusables for")
    parser.add_argument("--openai-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--skip-confusables", action="store_true", help="Skip confusable generation")
    parser.add_argument("--skip-negatives", action="store_true", help="Skip negative generation")
    parser.add_argument("--skip-silence", action="store_true", help="Skip silence generation")

    args = parser.parse_args()

    # Get API key
    openai_key = args.openai_key or os.environ.get("OPENAI_API_KEY")

    if not openai_key and not (args.skip_confusables and args.skip_negatives):
        print("ERROR: OpenAI API key required")
        print("  Set OPENAI_API_KEY environment variable or use --openai-key")
        sys.exit(1)

    print("=" * 60)
    print("Negative Sample Generator")
    print("=" * 60)
    print(f"Wake word: {args.wake_word}")
    print(f"Data directory: {DATA_DIR}")

    total_generated = 0

    # Generate confusable samples
    if not args.skip_confusables:
        confusable_dir = DATA_DIR / "confusable"
        count = generate_confusable_samples(args.wake_word, openai_key, confusable_dir)
        total_generated += count
        print(f"  Total confusable samples: {count}")

    # Generate negative samples
    if not args.skip_negatives:
        negative_dir = DATA_DIR / "negative"
        count = generate_negative_samples(openai_key, negative_dir)
        total_generated += count
        print(f"  Total negative samples: {count}")

    # Generate silence samples
    if not args.skip_silence:
        negative_dir = DATA_DIR / "negative"
        count = generate_silence_samples(negative_dir)
        total_generated += count

    print(f"\n{'=' * 60}")
    print(f"Done! Generated {total_generated} total samples")
    print(f"{'=' * 60}")
    print("\nDirectory structure:")
    print(f"  data/positive/     - Wake word samples")
    print(f"  data/negative/     - General speech and silence")
    print(f"  data/confusable/   - Similar-sounding words (hard negatives)")


if __name__ == "__main__":
    main()
