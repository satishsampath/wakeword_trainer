#!/usr/bin/env python3
"""
Simple CNN-based Wake Word Training Script

This script trains a standalone wake word model using a CNN architecture.
Unlike train_openwakeword.py, this does NOT require OpenWakeWord's embedding models.

The model takes raw audio as input and outputs a confidence score.

Requirements:
    pip install torch torchaudio

Usage:
    1. Edit config.py to set your WAKE_WORD and paths
    2. Place positive samples in data/positive/
    3. Place negative samples in data/negative/
    4. Run: python train_simple.py
"""

import os
import sys
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchaudio
import numpy as np
from pathlib import Path

# Import configuration
from config import (
    WAKE_WORD, MODEL_NAME, DATA_DIR, OUTPUT_DIR,
    SAMPLE_RATE, BATCH_SIZE, LEARNING_RATE
)

# Simple training uses fewer epochs by default
EPOCHS = 50


class WakeWordDataset(Dataset):
    """Dataset for wake word audio samples."""

    def __init__(self, positive_dir, negative_dir, transform=None):
        self.samples = []
        self.labels = []
        self.transform = transform

        # Load positive samples
        for wav_file in glob.glob(str(positive_dir / "*.wav")):
            self.samples.append(wav_file)
            self.labels.append(1)

        # Load negative samples
        for wav_file in glob.glob(str(negative_dir / "*.wav")):
            self.samples.append(wav_file)
            self.labels.append(0)

        print(f"Loaded {sum(self.labels)} positive, {len(self.labels) - sum(self.labels)} negative samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.samples[idx])

        # Resample if needed
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Pad or trim to fixed length (1.5 seconds)
        target_length = int(SAMPLE_RATE * 1.5)
        if waveform.shape[1] < target_length:
            padding = target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        else:
            waveform = waveform[:, :target_length]

        if self.transform:
            waveform = self.transform(waveform)

        return waveform.squeeze(0), self.labels[idx]


class SimpleWakeWordModel(nn.Module):
    """Simple CNN for wake word detection."""

    def __init__(self):
        super().__init__()

        # Mel spectrogram transform
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=512,
            hop_length=160,
            n_mels=96
        )

        # CNN layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        # Calculate FC input size based on mel spectrogram output
        # With 1.5s audio at 16kHz, hop_length=160: ~150 frames
        # After 3 pooling layers with n_mels=96: 96/8=12, 150/8~18
        self.fc1 = nn.Linear(128 * 12 * 18, 256)
        self.fc2 = nn.Linear(256, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, samples)

        # Compute mel spectrogram
        x = self.mel(x)  # (batch, n_mels, time)
        x = x.unsqueeze(1)  # (batch, 1, n_mels, time)

        # CNN
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        # FC layers
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))

        return x


def train_model():
    print(f"Training wake word model for: {WAKE_WORD}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check for data
    positive_dir = DATA_DIR / "positive"
    negative_dir = DATA_DIR / "negative"

    if not positive_dir.exists():
        print(f"ERROR: Positive samples directory not found: {positive_dir}")
        sys.exit(1)

    if not negative_dir.exists():
        print(f"WARNING: Negative samples directory not found: {negative_dir}")
        print("Creating empty negative directory...")
        negative_dir.mkdir(parents=True)

    # Create dataset
    dataset = WakeWordDataset(positive_dir, negative_dir)

    if len(dataset) == 0:
        print("ERROR: No training samples found!")
        sys.exit(1)

    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SimpleWakeWordModel().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    best_val_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.float().to(device)

            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                outputs = model(batch_x).squeeze()
                predicted = (outputs > 0.5).long()
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()

        val_acc = val_correct / val_total if val_total > 0 else 0

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {train_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), OUTPUT_DIR / f"{MODEL_NAME}.pt")
            print(f"  Saved best model with accuracy: {val_acc:.4f}")

    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")

    # Export to ONNX
    print("\nExporting to ONNX format...")
    model.load_state_dict(torch.load(OUTPUT_DIR / f"{MODEL_NAME}.pt"))
    model.eval()

    dummy_input = torch.randn(1, int(SAMPLE_RATE * 1.5)).to(device)
    onnx_path = OUTPUT_DIR / f"{MODEL_NAME}.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        input_names=['audio'],
        output_names=['confidence'],
        dynamic_axes={
            'audio': {0: 'batch_size'},
            'confidence': {0: 'batch_size'}
        }
    )

    print(f"ONNX model exported to: {onnx_path}")
    print("\nDone!")


if __name__ == "__main__":
    train_model()
