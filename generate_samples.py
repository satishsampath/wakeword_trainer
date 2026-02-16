#!/usr/bin/env python3
"""
Generate all training samples (positive, confusable, negative) for a wake word.

This is the single script to run when changing the wake word. It generates:
  1. Positive samples - the wake phrase in multiple voices/speeds
  2. Confusable samples - partial matches and similar phrases (hard negatives)
  3. General negatives - common phrases and silence

Usage:
    python generate_samples.py --wake-word "listen up tackle"
    python generate_samples.py --wake-word "okay computer start" --openai-key YOUR_KEY
    python generate_samples.py --wake-word "listen up tackle" --positive-only
    python generate_samples.py --wake-word "listen up tackle" --confusable-only

API key is read from (in order):
  1. --openai-key CLI argument
  2. OPENAI_API_KEY environment variable
  3. user_config.json file (key: "openai_api_key")
"""

import os
import sys
import json
import argparse
from pathlib import Path

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import torchaudio
    import torch
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False

DATA_DIR = Path(__file__).parent / "data"
CONFIG_FILE = Path(__file__).parent / "user_config.json"

VOICES = ["alloy", "ash", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer"]
SPEEDS = [0.90, 0.95, 1.00, 1.05, 1.10]

COMMON_PREFIXES = ["hey", "oh", "go", "the", "that", "a", "my", "ok", "so", "do"]

COMMON_PHRASES = [
    "hello", "hi there", "good morning", "good night",
    "how are you", "what time is it", "thank you", "you're welcome",
    "yes please", "no thanks", "sounds good", "I don't know",
    "play some music", "turn on the lights", "set a timer",
    "what's the weather", "call mom", "send a message",
    "open the app", "take a picture", "stop", "cancel",
    "go ahead", "never mind", "excuse me", "I'm sorry",
    "see you later", "have a good day", "that's fine", "absolutely",
]


def get_openai_key(args):
    """Get OpenAI API key from CLI arg, env var, or config file."""
    if args.openai_key:
        return args.openai_key
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                config = json.load(f)
            key = config.get("openai_api_key")
            if key:
                return key
        except Exception:
            pass
    return None


def convert_mp3_to_wav(mp3_path, wav_path):
    """Convert MP3 to 16kHz mono WAV using pydub."""
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_mp3(str(mp3_path))
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(str(wav_path), format="wav")
        mp3_path.unlink()
        return True
    except Exception as e:
        print(f"    Conversion error: {e}")
        return False


def generate_tts(client, text, output_path, voice="alloy", speed=1.0):
    """Generate a single TTS WAV file via OpenAI."""
    try:
        response = client.audio.speech.create(
            model="tts-1-hd",
            voice=voice,
            input=text,
            speed=speed
        )
        mp3_path = output_path.with_suffix(".mp3")
        response.stream_to_file(str(mp3_path))
        if not convert_mp3_to_wav(mp3_path, output_path):
            print(f"    Warning: keeping MP3 (conversion failed)")
        return True
    except Exception as e:
        print(f"    Error generating '{text}': {e}")
        return False


def build_confusable_phrases(wake_word):
    """
    Auto-generate confusable phrases from the wake word.

    For "listen up tackle", generates:
      - Individual words: "listen", "up", "tackle"
      - Last word with wrong prefixes: "hey tackle", "oh tackle", etc.
      - Partial phrases: "up tackle", "listen up", "listen tackle"
      - First word alone, last word alone
      - Two-word subsets
    """
    words = wake_word.lower().split()
    phrases = set()

    # Each individual word
    for word in words:
        phrases.add(word)

    if len(words) >= 2:
        # Last word with common prefixes
        last_word = words[-1]
        for prefix in COMMON_PREFIXES:
            if prefix != words[0]:
                phrases.add(f"{prefix} {last_word}")

        # All contiguous sub-phrases (shorter than the full phrase)
        for start in range(len(words)):
            for end in range(start + 1, len(words) + 1):
                sub = " ".join(words[start:end])
                if sub != wake_word.lower():
                    phrases.add(sub)

        # Skip one word from the middle (for 3+ word phrases)
        if len(words) >= 3:
            for skip in range(len(words)):
                sub_words = [w for i, w in enumerate(words) if i != skip]
                sub = " ".join(sub_words)
                if sub != wake_word.lower():
                    phrases.add(sub)

    # Remove the wake word itself if it snuck in
    phrases.discard(wake_word.lower())

    return sorted(phrases)


def generate_positive_samples(client, wake_word, output_dir):
    """Generate positive samples: wake word in all voice/speed combinations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(VOICES) * len(SPEEDS)
    count = 0
    skipped = 0

    print(f"\nGenerating {total} positive samples for: \"{wake_word}\"")
    print(f"  Output: {output_dir}")

    for voice in VOICES:
        for speed in SPEEDS:
            filename = f"positive_{voice}_{speed:.2f}.wav"
            output_path = output_dir / filename

            if output_path.exists():
                skipped += 1
                continue

            count += 1
            print(f"  [{count + skipped}/{total}] \"{wake_word}\" ({voice}, {speed:.2f}x)")
            generate_tts(client, wake_word, output_path, voice, speed)

    print(f"  Generated: {count}, Skipped (existing): {skipped}")
    return count


def generate_confusable_samples(client, wake_word, output_dir):
    """Generate confusable samples: partial matches and similar phrases."""
    output_dir.mkdir(parents=True, exist_ok=True)
    phrases = build_confusable_phrases(wake_word)

    total = len(phrases) * len(VOICES) * len(SPEEDS)
    count = 0
    skipped = 0

    print(f"\nGenerating confusable samples for: \"{wake_word}\"")
    print(f"  Confusable phrases ({len(phrases)}):")
    for p in phrases:
        print(f"    - \"{p}\"")
    print(f"  Combinations: {len(phrases)} phrases x {len(VOICES)} voices x {len(SPEEDS)} speeds = {total}")
    print(f"  Output: {output_dir}")

    for phrase in phrases:
        safe_name = "".join(c if c.isalnum() else "_" for c in phrase)[:30]
        for voice in VOICES:
            for speed in SPEEDS:
                filename = f"confusable_{safe_name}_{voice}_{speed:.2f}.wav"
                output_path = output_dir / filename

                if output_path.exists():
                    skipped += 1
                    continue

                count += 1
                if count % 10 == 1:
                    print(f"  [{count + skipped}/{total}] \"{phrase}\" ({voice}, {speed:.2f}x)")
                generate_tts(client, phrase, output_path, voice, speed)

    print(f"  Generated: {count}, Skipped (existing): {skipped}")
    return count


def generate_negative_samples(client, output_dir):
    """Generate general negative samples: common phrases in varied voices."""
    output_dir.mkdir(parents=True, exist_ok=True)
    voices = VOICES[:6]  # Use 6 voices for negatives
    total = len(COMMON_PHRASES) * len(voices)
    count = 0
    skipped = 0

    print(f"\nGenerating {total} general negative samples")
    print(f"  Output: {output_dir}")

    for phrase in COMMON_PHRASES:
        safe_name = "".join(c if c.isalnum() else "_" for c in phrase)[:30]
        for voice in voices:
            filename = f"negative_{safe_name}_{voice}.wav"
            output_path = output_dir / filename

            if output_path.exists():
                skipped += 1
                continue

            count += 1
            if count % 10 == 1:
                print(f"  [{count + skipped}/{total}] \"{phrase}\" ({voice})")
            generate_tts(client, phrase, output_path, voice)

    print(f"  Generated: {count}, Skipped (existing): {skipped}")
    return count


def generate_silence_samples(output_dir, num_samples=10):
    """Generate silence/near-silence WAV samples."""
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    print(f"\nGenerating {num_samples} silence samples")
    print(f"  Output: {output_dir}")

    if not HAS_TORCHAUDIO:
        print("  Warning: torchaudio not available, skipping silence generation")
        return 0

    for i in range(num_samples):
        output_path = output_dir / f"silence_{i:03d}.wav"
        if output_path.exists():
            continue

        # 2 seconds of near-silence with varying noise levels
        samples = 16000 * 2
        noise_level = 0.001 * (i + 1) / num_samples
        audio = torch.randn(1, samples) * noise_level
        torchaudio.save(str(output_path), audio, 16000)
        count += 1

    print(f"  Generated: {count}")
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Generate all training samples for a wake word"
    )
    parser.add_argument("--wake-word", required=True,
                        help="The wake word phrase (e.g., \"listen up tackle\")")
    parser.add_argument("--openai-key",
                        help="OpenAI API key")
    parser.add_argument("--positive-only", action="store_true",
                        help="Only generate positive samples")
    parser.add_argument("--confusable-only", action="store_true",
                        help="Only generate confusable samples")
    parser.add_argument("--negative-only", action="store_true",
                        help="Only generate negative + silence samples")
    parser.add_argument("--list-confusables", action="store_true",
                        help="Just print confusable phrases and exit (no generation)")

    args = parser.parse_args()

    # Just list confusables and exit
    if args.list_confusables:
        phrases = build_confusable_phrases(args.wake_word)
        print(f"Confusable phrases for \"{args.wake_word}\" ({len(phrases)}):")
        for p in phrases:
            print(f"  - \"{p}\"")
        return

    if not HAS_OPENAI:
        print("ERROR: openai package required. Install with: pip install openai")
        sys.exit(1)

    openai_key = get_openai_key(args)
    if not openai_key:
        print("ERROR: OpenAI API key required")
        print("  Use --openai-key, set OPENAI_API_KEY env var,")
        print(f"  or add \"openai_api_key\" to {CONFIG_FILE}")
        sys.exit(1)

    client = OpenAI(api_key=openai_key)

    # Determine which sample types to generate
    generate_all = not (args.positive_only or args.confusable_only or args.negative_only)

    print("=" * 60)
    print(f"Wake Word Sample Generator")
    print("=" * 60)
    print(f"Wake word: \"{args.wake_word}\"")
    print(f"Data directory: {DATA_DIR}")

    total = 0

    if generate_all or args.positive_only:
        total += generate_positive_samples(
            client, args.wake_word, DATA_DIR / "positive")

    if generate_all or args.confusable_only:
        total += generate_confusable_samples(
            client, args.wake_word, DATA_DIR / "confusable")

    if generate_all or args.negative_only:
        total += generate_negative_samples(
            client, DATA_DIR / "negative")
        total += generate_silence_samples(
            DATA_DIR / "negative")

    print(f"\n{'=' * 60}")
    print(f"Done! Generated {total} new samples")
    print(f"{'=' * 60}")

    # Print summary of data directories
    for subdir in ["positive", "confusable", "negative"]:
        d = DATA_DIR / subdir
        if d.exists():
            wav_count = len(list(d.glob("*.wav")))
            print(f"  {subdir}/: {wav_count} WAV files")


if __name__ == "__main__":
    main()
