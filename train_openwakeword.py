#!/usr/bin/env python3
"""
Train an OpenWakeWord-compatible wake word model.

This script trains a model that takes embedding vectors as input (not raw audio),
making it compatible with the OpenWakeWord pipeline:

    Raw Audio → Mel Spectrogram → Embedding Model → [Wake Word Classifier] → Confidence

The classifier expects:
- Input: [batch, 16, 96] - 16 frames of 96-dim embeddings
- Output: [batch, 1] - confidence score (0-1)

For short wake word clips, we:
1. Pad/tile embeddings to get at least 16 frames
2. Center the wake word embeddings within the 16-frame window
3. Use various padding strategies to augment the data

Requirements:
    pip install torch torchaudio onnxruntime numpy

Usage:
    1. Edit config.py to set your WAKE_WORD and paths
    2. Place positive samples in data/positive/
    3. Place negative samples in data/negative/
    4. Run: python train_openwakeword.py
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
import random

# Import configuration
from config import (
    WAKE_WORD, MODEL_NAME, DATA_DIR, OUTPUT_DIR,
    MEL_MODEL_PATH, EMBEDDING_MODEL_PATH,
    SAMPLE_RATE, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    SAMPLES_PER_FRAME, MEL_BANDS, EMBEDDING_SIZE, FEATURE_FRAMES
)

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

        # Apply normalization
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

    return result


class WakeWordDataset(Dataset):
    """Dataset for wake word training with proper embedding handling."""

    def __init__(self, positive_files, negative_files, embedder, augment=True):
        """
        Args:
            positive_files: List of paths to positive audio files
            negative_files: List of paths to negative audio files
            embedder: OpenWakeWordEmbedder instance
            augment: Whether to augment data with different padding strategies
        """
        self.samples = []
        self.labels = []
        self.embedder = embedder
        self.augment = augment

        print(f"Processing {len(positive_files)} positive files...")
        for i, wav_file in enumerate(positive_files):
            if i % 500 == 0:
                print(f"  {i}/{len(positive_files)}...")

            try:
                embeddings = self._load_and_embed(wav_file)
                if len(embeddings) > 0:
                    # Create multiple samples with different padding strategies
                    if augment:
                        for strategy in ['center', 'repeat', 'random']:
                            padded = pad_embeddings_to_frames(embeddings, strategy=strategy)
                            self.samples.append(padded)
                            self.labels.append(1)
                    else:
                        padded = pad_embeddings_to_frames(embeddings, strategy='center')
                        self.samples.append(padded)
                        self.labels.append(1)
            except Exception as e:
                pass  # Skip failed files

        print(f"Processing {len(negative_files)} negative files...")
        for i, wav_file in enumerate(negative_files):
            if i % 200 == 0:
                print(f"  {i}/{len(negative_files)}...")

            try:
                embeddings = self._load_and_embed(wav_file)
                if len(embeddings) > 0:
                    if augment:
                        for strategy in ['center', 'repeat', 'random']:
                            padded = pad_embeddings_to_frames(embeddings, strategy=strategy)
                            self.samples.append(padded)
                            self.labels.append(0)
                    else:
                        padded = pad_embeddings_to_frames(embeddings, strategy='center')
                        self.samples.append(padded)
                        self.labels.append(0)
            except Exception as e:
                pass

        # Balance dataset if needed
        pos_count = sum(self.labels)
        neg_count = len(self.labels) - pos_count
        print(f"\nDataset: {pos_count} positive, {neg_count} negative samples")

    def _load_and_embed(self, wav_path):
        """Load audio file and extract embeddings."""
        waveform, sr = torchaudio.load(wav_path)

        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        audio = waveform.squeeze().numpy()

        # Pad to minimum length for embedding extraction
        min_samples = int(SAMPLE_RATE * 2.0)
        if len(audio) < min_samples:
            audio = np.pad(audio, (0, min_samples - len(audio)))

        return self.embedder.extract_embeddings(audio)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.samples[idx]), self.labels[idx]


class WakeWordClassifier(nn.Module):
    """
    RNN classifier for wake word detection.

    Input: [batch, 16, 96] - 16 frames of 96-dim embeddings
    Output: [batch, 1] - confidence score
    """

    def __init__(self, input_size=EMBEDDING_SIZE, hidden_size=64, num_layers=2):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [batch, 16, 96]
        gru_out, _ = self.gru(x)
        last_out = gru_out[:, -1, :]
        return self.fc(last_out)


def train_model():
    """Main training function."""
    print(f"Training OpenWakeWord-compatible model for: {WAKE_WORD}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

    # Get audio files
    positive_dir = DATA_DIR / "positive"
    negative_dir = DATA_DIR / "negative"

    positive_files = list(positive_dir.glob("*.wav"))
    negative_files = list(negative_dir.glob("*.wav"))

    print(f"\nFound {len(positive_files)} positive, {len(negative_files)} negative audio files")

    if len(positive_files) == 0:
        print("ERROR: No positive samples found!")
        sys.exit(1)

    # Create dataset
    print("\n=== Building Dataset ===")
    dataset = WakeWordDataset(positive_files, negative_files, embedder, augment=True)

    if len(dataset) == 0:
        print("ERROR: No training samples created!")
        sys.exit(1)

    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    print(f"\nTrain: {train_size}, Validation: {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0)

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = WakeWordClassifier().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Training loop
    print("\n=== Training ===")
    best_val_acc = 0
    patience_counter = 0
    max_patience = 15

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.float().to(device)

            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (outputs > 0.5).long()
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y.long()).sum().item()

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0

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

        train_acc = train_correct / train_total if train_total > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), OUTPUT_DIR / f"{MODEL_NAME}_classifier.pt")
            print(f"  * Saved best model (acc: {val_acc:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        if val_acc >= 0.995 and epoch >= 30:
            print("Achieved excellent accuracy, stopping")
            break

    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")

    # Export to ONNX
    print("\n=== Exporting to ONNX ===")
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

    # Verify
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
        "best_accuracy": float(best_val_acc),
        "description": "OpenWakeWord-compatible wake word classifier"
    }

    with open(OUTPUT_DIR / f"{MODEL_NAME}_info.json", 'w') as f:
        json.dump(model_info, f, indent=2)

    print(f"\n{'='*50}")
    print("DONE!")
    print(f"{'='*50}")
    print(f"\nModel: {onnx_path}")
    print(f"Size: {onnx_path.stat().st_size / 1024:.1f} KB")
    print(f"\nTo use with OpenWakeWord, copy to your models directory.")


if __name__ == "__main__":
    train_model()
