#!/usr/bin/env python3
"""
Train an OpenWakeWord-compatible wake word model.

This script trains a model that takes embedding vectors as input (not raw audio),
making it compatible with the OpenWakeWord pipeline:

    Raw Audio -> Mel Spectrogram -> Embedding Model -> [Wake Word Classifier] -> Confidence

The classifier expects:
- Input: [batch, 16, 96] - 16 frames of 96-dim embeddings
- Output: [batch, 1] - confidence score (0-1)

For high-quality wake word detection, we:
1. Use diverse positive samples (multiple TTS voices, real recordings)
2. Use hard negatives (confusable words that sound similar)
3. Use general negatives (common speech, background noise)
4. Apply robust augmentation (noise, reverb, pitch shift, etc.)

Requirements:
    pip install torch torchaudio onnxruntime numpy scipy

Usage:
    1. Edit config.py to set your WAKE_WORD and paths
    2. Place positive samples in data/positive/
    3. Place negative samples in data/negative/
    4. Optionally place confusable samples in data/confusable/
    5. Run: python train_openwakeword.py
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import numpy as np
from pathlib import Path
import random
from collections import Counter

# Import configuration
from config import (
    WAKE_WORD, MODEL_NAME, DATA_DIR, OUTPUT_DIR,
    MEL_MODEL_PATH, EMBEDDING_MODEL_PATH,
    SAMPLE_RATE, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    SAMPLES_PER_FRAME, MEL_BANDS, EMBEDDING_SIZE, FEATURE_FRAMES
)

# Import augmentation
try:
    from augmentation import AudioAugmenter, generate_confusable_words
    HAS_AUGMENTATION = True
except ImportError:
    HAS_AUGMENTATION = False

# Try to import ONNX runtime for embedding extraction
try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    print("Warning: onnxruntime not installed. Install with: pip install onnxruntime")

# Try to import torchaudio for audio loading
try:
    import torchaudio
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False
    print("Warning: torchaudio not installed.")


class OpenWakeWordEmbedder:
    """Extract embeddings using OpenWakeWord's preprocessing pipeline."""

    def __init__(self, mel_model_path, embedding_model_path):
        if not HAS_ONNX:
            raise RuntimeError("onnxruntime required for embedding extraction")

        print(f"Loading mel model from: {mel_model_path}")
        self.mel_session = ort.InferenceSession(str(mel_model_path))
        self.mel_input_name = self.mel_session.get_inputs()[0].name

        print(f"Loading embedding model from: {embedding_model_path}")
        self.embedding_session = ort.InferenceSession(str(embedding_model_path))
        self.embedding_input_name = self.embedding_session.get_inputs()[0].name

    def extract_embeddings(self, audio_samples):
        """
        Extract embeddings from audio samples.

        Args:
            audio_samples: numpy array of audio samples (16kHz, mono, float32 normalized)

        Returns:
            numpy array of embeddings [num_embeddings, 96]
        """
        # Process audio in 80ms chunks to build mel buffer
        mel_buffer = []

        for i in range(0, len(audio_samples) - SAMPLES_PER_FRAME + 1, SAMPLES_PER_FRAME):
            chunk = audio_samples[i:i + SAMPLES_PER_FRAME]
            mel_output = self._compute_mel(chunk)

            # Each mel output has multiple frames
            for j in range(mel_output.shape[0]):
                mel_buffer.append(mel_output[j])

        # Generate embeddings with stride of 8 mel frames
        # Need 76 mel frames for one embedding
        STRIDE = 8
        embeddings = []

        for window_start in range(0, len(mel_buffer) - 76 + 1, STRIDE):
            mel_window = np.array(mel_buffer[window_start:window_start + 76])
            embedding = self._compute_embedding(mel_window)
            embeddings.append(embedding)

        return np.array(embeddings) if embeddings else np.zeros((0, EMBEDDING_SIZE))

    def _compute_mel(self, samples):
        """Compute mel spectrogram for one audio chunk."""
        input_data = samples.reshape(1, -1).astype(np.float32)
        outputs = self.mel_session.run(None, {self.mel_input_name: input_data})
        output = outputs[0]

        if output.ndim == 4:
            output = output[0, 0]
        elif output.ndim == 3:
            output = output[0]

        # Apply normalization (matching OpenWakeWord)
        output = (output / 10.0) + 2.0
        return output

    def _compute_embedding(self, mel_frames):
        """Compute embedding from 76 mel frames."""
        input_data = mel_frames.reshape(1, 76, MEL_BANDS, 1).astype(np.float32)
        outputs = self.embedding_session.run(None, {self.embedding_input_name: input_data})
        output = outputs[0]
        return output.flatten()[:EMBEDDING_SIZE]


def pad_embeddings_to_frames(embeddings, target_frames=FEATURE_FRAMES, strategy='center'):
    """
    Pad embeddings to reach target_frames.

    Strategies:
    - 'center': Center the embeddings and pad with zeros
    - 'repeat': Repeat embeddings to fill the window
    - 'random': Place embeddings at random position with zero padding
    - 'left': Align to left with zero padding
    - 'right': Align to right with zero padding
    """
    n = len(embeddings)

    if n >= target_frames:
        # If we have enough, return centered slice
        start = (n - target_frames) // 2
        return embeddings[start:start + target_frames]

    if n == 0:
        return np.zeros((target_frames, EMBEDDING_SIZE))

    result = np.zeros((target_frames, EMBEDDING_SIZE))

    if strategy == 'center':
        start = (target_frames - n) // 2
        result[start:start + n] = embeddings

    elif strategy == 'repeat':
        # Tile embeddings to fill
        idx = 0
        for i in range(target_frames):
            result[i] = embeddings[idx % n]
            idx += 1

    elif strategy == 'random':
        max_start = target_frames - n
        start = random.randint(0, max_start)
        result[start:start + n] = embeddings

    elif strategy == 'left':
        result[:n] = embeddings

    elif strategy == 'right':
        result[-n:] = embeddings

    return result


class WakeWordDataset(Dataset):
    """
    Enhanced dataset for wake word training.

    Supports:
    - Positive samples (wake word)
    - Negative samples (general speech)
    - Confusable samples (similar-sounding words, treated as hard negatives)
    - Advanced augmentation
    """

    def __init__(self, positive_files, negative_files, confusable_files,
                 embedder, augmenter=None, augment_factor=5):
        """
        Args:
            positive_files: List of paths to positive audio files
            negative_files: List of paths to negative audio files
            confusable_files: List of paths to confusable audio files (hard negatives)
            embedder: OpenWakeWordEmbedder instance
            augmenter: AudioAugmenter instance (optional)
            augment_factor: How many augmented versions to create per sample
        """
        self.samples = []
        self.labels = []
        self.embedder = embedder
        self.augmenter = augmenter
        self.augment_factor = augment_factor

        # Process positive samples
        print(f"\nProcessing {len(positive_files)} positive files...")
        for i, wav_file in enumerate(positive_files):
            if i % 100 == 0 and i > 0:
                print(f"  {i}/{len(positive_files)}...")

            self._process_file(wav_file, label=1, is_positive=True)

        # Process confusable samples (hard negatives)
        if confusable_files:
            print(f"\nProcessing {len(confusable_files)} confusable files (hard negatives)...")
            for i, wav_file in enumerate(confusable_files):
                if i % 100 == 0 and i > 0:
                    print(f"  {i}/{len(confusable_files)}...")

                self._process_file(wav_file, label=0, is_positive=False)

        # Process negative samples
        print(f"\nProcessing {len(negative_files)} negative files...")
        for i, wav_file in enumerate(negative_files):
            if i % 200 == 0 and i > 0:
                print(f"  {i}/{len(negative_files)}...")

            self._process_file(wav_file, label=0, is_positive=False)

        # Print dataset statistics
        pos_count = sum(self.labels)
        neg_count = len(self.labels) - pos_count
        print(f"\nDataset: {pos_count} positive, {neg_count} negative samples")

        # Warn about imbalance
        if pos_count > 0 and neg_count > 0:
            ratio = max(pos_count, neg_count) / min(pos_count, neg_count)
            if ratio > 3:
                print(f"WARNING: Dataset is imbalanced ({ratio:.1f}:1). "
                      f"Consider adding more {'negatives' if pos_count > neg_count else 'positives'}.")

    def _process_file(self, wav_path, label, is_positive):
        """Process a single audio file and add to dataset."""
        try:
            audio = self._load_audio(wav_path)
            if audio is None:
                return

            # Create multiple augmented versions
            if self.augmenter and self.augment_factor > 1:
                # Always include original with center padding
                embeddings = self.embedder.extract_embeddings(audio)
                if len(embeddings) > 0:
                    padded = pad_embeddings_to_frames(embeddings, strategy='center')
                    self.samples.append(padded)
                    self.labels.append(label)

                # Add augmented versions
                for _ in range(self.augment_factor - 1):
                    intensity = random.choice(['light', 'medium', 'heavy'])
                    aug_audio, _ = self.augmenter.random_augment(audio, intensity)

                    embeddings = self.embedder.extract_embeddings(aug_audio)
                    if len(embeddings) > 0:
                        # Random padding strategy for variety
                        strategy = random.choice(['center', 'random', 'left', 'right'])
                        padded = pad_embeddings_to_frames(embeddings, strategy=strategy)
                        self.samples.append(padded)
                        self.labels.append(label)
            else:
                # No augmentation - use multiple padding strategies
                embeddings = self.embedder.extract_embeddings(audio)
                if len(embeddings) > 0:
                    strategies = ['center', 'random', 'left', 'right', 'repeat']
                    for strategy in strategies[:3]:  # Use 3 strategies
                        padded = pad_embeddings_to_frames(embeddings, strategy=strategy)
                        self.samples.append(padded)
                        self.labels.append(label)

        except Exception as e:
            pass  # Skip failed files silently

    def _load_audio(self, wav_path):
        """Load and preprocess audio file."""
        try:
            import soundfile as sf
            from scipy import signal

            audio, sr = sf.read(wav_path, dtype='float32')

            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            # Resample if needed
            if sr != SAMPLE_RATE:
                num_samples = int(len(audio) * SAMPLE_RATE / sr)
                audio = signal.resample(audio, num_samples)

            # Normalize
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val * 0.9

            # Pad to minimum length for embedding extraction
            min_samples = int(SAMPLE_RATE * 2.0)
            if len(audio) < min_samples:
                audio = np.pad(audio, (0, min_samples - len(audio)))

            return audio
        except Exception:
            return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.samples[idx]), self.labels[idx]


class WakeWordClassifier(nn.Module):
    """
    Enhanced RNN classifier for wake word detection.

    Input: [batch, 16, 96] - 16 frames of 96-dim embeddings
    Output: [batch, 1] - confidence score

    Architecture improvements:
    - Bidirectional GRU for better temporal modeling
    - Layer normalization for training stability
    - Residual connections
    """

    def __init__(self, input_size=EMBEDDING_SIZE, hidden_size=64, num_layers=2):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)

        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [batch, 16, 96]

        # Project input
        x = self.input_proj(x)  # [batch, 16, hidden]

        # GRU
        gru_out, _ = self.gru(x)  # [batch, 16, hidden*2]

        # Use last output + apply layer norm
        last_out = self.layer_norm(gru_out[:, -1, :])

        return self.fc(last_out)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Focuses training on hard examples.
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


def train_model():
    """Main training function with enhanced pipeline."""
    print("=" * 60)
    print(f"OpenWakeWord Training: {WAKE_WORD}")
    print("=" * 60)
    print(f"\nData directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check dependencies
    if not HAS_ONNX or not HAS_TORCHAUDIO:
        print("\nERROR: Required dependencies missing!")
        print("Install with: pip install onnxruntime torchaudio soundfile")
        sys.exit(1)

    if not MEL_MODEL_PATH.exists() or not EMBEDDING_MODEL_PATH.exists():
        print(f"ERROR: OpenWakeWord models not found!")
        print(f"  Expected mel model: {MEL_MODEL_PATH}")
        print(f"  Expected embedding model: {EMBEDDING_MODEL_PATH}")
        print(f"\nDownload these models from:")
        print(f"  https://github.com/dscripka/openWakeWord/tree/main/openwakeword/resources/models")
        sys.exit(1)

    # Initialize embedder
    embedder = OpenWakeWordEmbedder(MEL_MODEL_PATH, EMBEDDING_MODEL_PATH)

    # Initialize augmenter if available
    augmenter = None
    if HAS_AUGMENTATION:
        augmenter = AudioAugmenter(SAMPLE_RATE)

        # Load background noises if available
        noise_dir = DATA_DIR / "background_noise"
        if noise_dir.exists():
            augmenter.load_background_noises(noise_dir)

    # Get audio files
    positive_dir = DATA_DIR / "positive"
    negative_dir = DATA_DIR / "negative"
    confusable_dir = DATA_DIR / "confusable"

    positive_files = list(positive_dir.glob("*.wav"))
    negative_files = list(negative_dir.glob("*.wav"))
    confusable_files = list(confusable_dir.glob("*.wav")) if confusable_dir.exists() else []

    print(f"\nFound audio files:")
    print(f"  Positive (wake word): {len(positive_files)}")
    print(f"  Negative (general): {len(negative_files)}")
    print(f"  Confusable (hard negative): {len(confusable_files)}")

    if len(positive_files) == 0:
        print("\nERROR: No positive samples found!")
        print(f"Please add .wav files to: {positive_dir}")
        sys.exit(1)

    if len(negative_files) == 0:
        print("\nWARNING: No negative samples found!")
        print("Model may have high false positive rate.")
        print(f"Add .wav files to: {negative_dir}")

    # Create dataset
    print("\n" + "=" * 40)
    print("Building Dataset")
    print("=" * 40)

    # Determine augmentation factor based on sample count
    total_raw_samples = len(positive_files) + len(negative_files) + len(confusable_files)
    if total_raw_samples < 100:
        augment_factor = 10
    elif total_raw_samples < 500:
        augment_factor = 5
    else:
        augment_factor = 3

    print(f"Augmentation factor: {augment_factor}x per sample")

    dataset = WakeWordDataset(
        positive_files, negative_files, confusable_files,
        embedder, augmenter, augment_factor=augment_factor
    )

    if len(dataset) == 0:
        print("ERROR: No training samples created!")
        sys.exit(1)

    # Split train/val
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    print(f"\nTrain: {train_size}, Validation: {val_size}")

    # Create data loaders with weighted sampling for imbalanced data
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0)

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = WakeWordClassifier().to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Use focal loss for imbalanced data
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # Training loop
    print("\n" + "=" * 40)
    print("Training")
    print("=" * 40)

    best_val_f1 = 0
    best_val_acc = 0
    patience_counter = 0
    max_patience = 20

    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        train_tp, train_fp, train_fn = 0, 0, 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.float().to(device)

            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            predicted = (outputs > 0.5).long()
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y.long()).sum().item()

            # Track TP, FP, FN for F1
            train_tp += ((predicted == 1) & (batch_y.long() == 1)).sum().item()
            train_fp += ((predicted == 1) & (batch_y.long() == 0)).sum().item()
            train_fn += ((predicted == 0) & (batch_y.long() == 1)).sum().item()

        scheduler.step()

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        val_tp, val_fp, val_fn, val_tn = 0, 0, 0, 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.float().to(device)

                outputs = model(batch_x).squeeze()
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

                predicted = (outputs > 0.5).long()
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y.long()).sum().item()

                val_tp += ((predicted == 1) & (batch_y.long() == 1)).sum().item()
                val_fp += ((predicted == 1) & (batch_y.long() == 0)).sum().item()
                val_fn += ((predicted == 0) & (batch_y.long() == 1)).sum().item()
                val_tn += ((predicted == 0) & (batch_y.long() == 0)).sum().item()

        # Calculate metrics
        train_acc = train_correct / train_total if train_total > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0

        # F1 score
        val_precision = val_tp / (val_tp + val_fp) if (val_tp + val_fp) > 0 else 0
        val_recall = val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 0
        val_f1 = 2 * val_precision * val_recall / (val_precision + val_recall) if (val_precision + val_recall) > 0 else 0

        # False positive rate
        val_fpr = val_fp / (val_fp + val_tn) if (val_fp + val_tn) > 0 else 0

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Loss: {avg_train_loss:.4f}/{avg_val_loss:.4f} | "
              f"Acc: {train_acc:.3f}/{val_acc:.3f} | "
              f"F1: {val_f1:.3f} | FPR: {val_fpr:.3f}")

        # Save best model (based on F1 score to balance precision/recall)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), OUTPUT_DIR / f"{MODEL_NAME}_classifier.pt")
            print(f"  -> Saved best model (F1: {val_f1:.4f}, Acc: {val_acc:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= max_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

        # Stop if excellent performance
        if val_f1 >= 0.98 and val_acc >= 0.98 and epoch >= 30:
            print("\nAchieved excellent performance, stopping")
            break

    print(f"\n{'=' * 40}")
    print("Training Complete!")
    print(f"{'=' * 40}")
    print(f"Best validation F1: {best_val_f1:.4f}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    # Export to ONNX
    print("\n" + "=" * 40)
    print("Exporting to ONNX")
    print("=" * 40)

    model.load_state_dict(torch.load(OUTPUT_DIR / f"{MODEL_NAME}_classifier.pt", map_location='cpu'))
    model = model.to('cpu')
    model.eval()

    dummy_input = torch.randn(1, FEATURE_FRAMES, EMBEDDING_SIZE)
    onnx_path = OUTPUT_DIR / f"{MODEL_NAME}.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"ONNX model exported to: {onnx_path}")

    # Verify ONNX model
    ort_session = ort.InferenceSession(str(onnx_path))
    test_input = np.random.randn(1, FEATURE_FRAMES, EMBEDDING_SIZE).astype(np.float32)

    with torch.no_grad():
        torch_output = model(torch.FloatTensor(test_input)).numpy()

    ort_output = ort_session.run(None, {ort_session.get_inputs()[0].name: test_input})[0]

    print(f"PyTorch output: {torch_output[0][0]:.6f}")
    print(f"ONNX output: {ort_output[0][0]:.6f}")
    print(f"Match: {np.allclose(torch_output, ort_output, atol=1e-5)}")

    # Save model info
    model_info = {
        "name": MODEL_NAME,
        "wake_word": WAKE_WORD,
        "input_shape": [1, FEATURE_FRAMES, EMBEDDING_SIZE],
        "output_shape": [1, 1],
        "best_f1_score": float(best_val_f1),
        "best_accuracy": float(best_val_acc),
        "training_samples": len(dataset),
        "description": "OpenWakeWord-compatible wake word classifier"
    }

    with open(OUTPUT_DIR / f"{MODEL_NAME}_info.json", 'w') as f:
        json.dump(model_info, f, indent=2)

    print(f"\n{'=' * 60}")
    print("DONE!")
    print(f"{'=' * 60}")
    print(f"\nModel: {onnx_path}")
    print(f"Size: {onnx_path.stat().st_size / 1024:.1f} KB")
    print(f"\nTo use with Nevyx, copy to:")
    print(f"  %USERPROFILE%\\.nevyx\\models\\{MODEL_NAME}.onnx")


if __name__ == "__main__":
    train_model()
