#!/usr/bin/env python3
"""
Advanced Audio Augmentation for Wake Word Training

Provides robust augmentation techniques to improve model generalization:
- Background noise mixing
- Room reverb simulation
- Pitch shifting and time stretching
- Dynamic range compression
- Low/high pass filtering
"""

import numpy as np
import random
from pathlib import Path
from typing import Optional, Tuple, List

# Try to import optional dependencies
try:
    import torch
    import torchaudio
    import torchaudio.functional as F
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False

try:
    import scipy.signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class AudioAugmenter:
    """Applies various augmentation techniques to audio samples."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.background_noises: List[np.ndarray] = []

    def load_background_noises(self, noise_dir: Path):
        """Load background noise files for mixing."""
        if not noise_dir.exists():
            return

        try:
            import soundfile as sf
            from scipy import signal
        except ImportError:
            print("Warning: soundfile/scipy not available for loading background noises")
            return

        for wav_file in noise_dir.glob("*.wav"):
            try:
                audio, sr = sf.read(str(wav_file), dtype='float32')
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)
                if sr != self.sample_rate:
                    num_samples = int(len(audio) * self.sample_rate / sr)
                    audio = signal.resample(audio, num_samples)
                self.background_noises.append(audio)
            except Exception:
                pass

        print(f"Loaded {len(self.background_noises)} background noise files")

    def add_gaussian_noise(self, audio: np.ndarray, snr_db: float = 20) -> np.ndarray:
        """Add Gaussian noise at specified SNR."""
        signal_power = np.mean(audio ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
        return audio + noise.astype(audio.dtype)

    def add_background_noise(self, audio: np.ndarray, snr_db: float = 15) -> np.ndarray:
        """Mix with random background noise."""
        if not self.background_noises:
            return self.add_gaussian_noise(audio, snr_db)

        noise = random.choice(self.background_noises)

        # Loop noise if too short
        if len(noise) < len(audio):
            repeats = int(np.ceil(len(audio) / len(noise)))
            noise = np.tile(noise, repeats)

        # Random start position
        start = random.randint(0, len(noise) - len(audio))
        noise_segment = noise[start:start + len(audio)]

        # Calculate mixing ratio for target SNR
        signal_power = np.mean(audio ** 2) + 1e-10
        noise_power = np.mean(noise_segment ** 2) + 1e-10
        scale = np.sqrt(signal_power / (noise_power * (10 ** (snr_db / 10))))

        return audio + noise_segment * scale

    def change_volume(self, audio: np.ndarray, gain_db: float) -> np.ndarray:
        """Change volume by specified dB."""
        gain = 10 ** (gain_db / 20)
        return np.clip(audio * gain, -1.0, 1.0)

    def add_reverb(self, audio: np.ndarray, room_size: float = 0.3,
                   damping: float = 0.5) -> np.ndarray:
        """Add simple reverb effect using comb filters."""
        if not HAS_SCIPY:
            return audio

        # Simple reverb using multiple delayed copies
        output = audio.copy()
        delays_ms = [23, 37, 43, 53, 67, 79]  # Prime number delays

        for i, delay_ms in enumerate(delays_ms):
            delay_samples = int(self.sample_rate * delay_ms / 1000 * room_size)
            decay = damping ** (i + 1) * 0.3

            if delay_samples < len(audio):
                delayed = np.zeros_like(audio)
                delayed[delay_samples:] = audio[:-delay_samples] * decay
                output += delayed

        # Normalize
        max_val = np.max(np.abs(output))
        if max_val > 1.0:
            output = output / max_val

        return output

    def time_stretch(self, audio: np.ndarray, rate: float) -> np.ndarray:
        """Time stretch without changing pitch (simple resampling approach)."""
        if not HAS_SCIPY:
            return audio

        # Simple approach: resample
        new_length = int(len(audio) / rate)
        indices = np.linspace(0, len(audio) - 1, new_length)
        stretched = np.interp(indices, np.arange(len(audio)), audio)

        return stretched.astype(audio.dtype)

    def pitch_shift(self, audio: np.ndarray, semitones: float) -> np.ndarray:
        """Shift pitch by specified semitones."""
        if not HAS_TORCHAUDIO:
            return audio

        try:
            waveform = torch.FloatTensor(audio).unsqueeze(0)

            # Use torchaudio's pitch shift if available
            if hasattr(F, 'pitch_shift'):
                shifted = F.pitch_shift(waveform, self.sample_rate, semitones)
                return shifted.squeeze().numpy()
        except Exception:
            pass

        return audio

    def lowpass_filter(self, audio: np.ndarray, cutoff_hz: float = 4000) -> np.ndarray:
        """Apply lowpass filter (simulates phone/low quality audio)."""
        if not HAS_SCIPY:
            return audio

        nyquist = self.sample_rate / 2
        normalized_cutoff = cutoff_hz / nyquist
        b, a = scipy.signal.butter(4, normalized_cutoff, btype='low')

        return scipy.signal.filtfilt(b, a, audio).astype(audio.dtype)

    def highpass_filter(self, audio: np.ndarray, cutoff_hz: float = 100) -> np.ndarray:
        """Apply highpass filter (removes low frequency rumble)."""
        if not HAS_SCIPY:
            return audio

        nyquist = self.sample_rate / 2
        normalized_cutoff = cutoff_hz / nyquist
        b, a = scipy.signal.butter(4, normalized_cutoff, btype='high')

        return scipy.signal.filtfilt(b, a, audio).astype(audio.dtype)

    def random_augment(self, audio: np.ndarray, intensity: str = 'medium') -> Tuple[np.ndarray, str]:
        """Apply random augmentation with specified intensity."""
        augmentations = []
        result = audio.copy()

        if intensity == 'light':
            prob = 0.3
            noise_range = (25, 35)
            vol_range = (-3, 3)
        elif intensity == 'medium':
            prob = 0.5
            noise_range = (15, 25)
            vol_range = (-6, 6)
        else:  # heavy
            prob = 0.7
            noise_range = (10, 20)
            vol_range = (-10, 10)

        # Random volume change
        if random.random() < prob:
            gain = random.uniform(*vol_range)
            result = self.change_volume(result, gain)
            augmentations.append(f"vol{gain:+.1f}dB")

        # Random noise
        if random.random() < prob:
            snr = random.uniform(*noise_range)
            if self.background_noises and random.random() < 0.5:
                result = self.add_background_noise(result, snr)
                augmentations.append(f"bg{snr:.0f}dB")
            else:
                result = self.add_gaussian_noise(result, snr)
                augmentations.append(f"noise{snr:.0f}dB")

        # Random reverb
        if random.random() < prob * 0.5:
            room_size = random.uniform(0.2, 0.6)
            result = self.add_reverb(result, room_size)
            augmentations.append(f"reverb{room_size:.1f}")

        # Random lowpass (phone quality)
        if random.random() < prob * 0.3:
            cutoff = random.uniform(3000, 5000)
            result = self.lowpass_filter(result, cutoff)
            augmentations.append(f"lp{cutoff:.0f}")

        # Random time stretch (slight)
        if random.random() < prob * 0.3:
            rate = random.uniform(0.9, 1.1)
            result = self.time_stretch(result, rate)
            augmentations.append(f"stretch{rate:.2f}")

        aug_str = "_".join(augmentations) if augmentations else "clean"
        return result, aug_str


def generate_confusable_words(wake_word: str) -> List[str]:
    """
    Generate phonetically similar words that could be confused with the wake word.
    These are used as hard negatives during training.
    """
    # Common confusable patterns
    confusables = []

    words = wake_word.lower().split()

    # For "Hey X" patterns
    if words[0] in ['hey', 'hi', 'hay']:
        prefixes = ['hey', 'hi', 'hay', 'say', 'day', 'may', 'way', 'pay', 'okay', 'a']

        if len(words) > 1:
            target = words[1]

            # Generate similar-sounding alternatives
            similar_names = {
                'nevyx': ['nexus', 'netflix', 'nervous', 'devin', 'kevin', 'devon', 'heaven', 'seven', 'levis'],
                'jarvis': ['service', 'harvest', 'carvis', 'marvis', 'jarvas', 'travis', 'jarred'],
                'alexa': ['alexis', 'alex', 'lexus', 'flexer', 'elixir'],
                'siri': ['series', 'sorry', 'surrey', 'sierra', 'cereal'],
                'computer': ['commuter', 'computes', 'come here', 'conductor'],
                'assistant': ['assistance', 'insistent', 'persistent', 'resistant'],
                'google': ['goggle', 'giggle', 'goober', 'noodle'],
            }

            # Get similar words for target
            similar = similar_names.get(target, [])

            # Generate combinations
            for prefix in prefixes:
                if prefix != words[0]:
                    confusables.append(f"{prefix} {target}")
                for sim in similar:
                    confusables.append(f"{prefix} {sim}")
                    confusables.append(f"{words[0]} {sim}")

            # Add some without prefix
            for sim in similar:
                confusables.append(sim)

    # Add common false trigger phrases
    general_confusables = [
        "okay",
        "hey there",
        "hey you",
        "hey now",
        "say what",
        "no way",
        "today",
        "anyway",
        "always",
        "never",
        "whatever",
        "however",
        "clever",
        "forever",
        "weather",
        "whether",
        "together",
        "remember",
        "november",
        "december",
    ]
    confusables.extend(general_confusables)

    # Remove duplicates and the actual wake word
    confusables = list(set(confusables))
    confusables = [c for c in confusables if c.lower() != wake_word.lower()]

    return confusables


def get_common_phrases() -> List[str]:
    """
    Get common spoken phrases for negative samples.
    These represent normal speech that should NOT trigger the wake word.
    """
    return [
        # Greetings
        "hello", "hi there", "good morning", "good afternoon", "good evening",
        "how are you", "nice to meet you", "goodbye", "see you later",

        # Common commands/questions
        "what time is it", "what's the weather", "play some music",
        "turn on the lights", "set a timer", "remind me", "call mom",
        "send a message", "search for", "open the door",

        # Everyday phrases
        "I don't know", "let me think", "that's interesting", "sounds good",
        "no problem", "thank you", "you're welcome", "excuse me",
        "I'm sorry", "please help", "can you", "would you",

        # Numbers and letters
        "one two three", "a b c", "first second third",

        # Filler words and sounds
        "um", "uh", "hmm", "yeah", "yes", "no", "okay", "alright",

        # Short phrases
        "go ahead", "come on", "let's go", "wait a minute", "hold on",
        "be right back", "just a second", "one moment", "hang on",
    ]
