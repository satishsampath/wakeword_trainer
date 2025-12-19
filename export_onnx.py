#!/usr/bin/env python3
"""
Export the trained wake word model to ONNX format.

Since torchaudio's MelSpectrogram uses STFT which isn't ONNX-compatible,
we export only the CNN classifier part. The mel spectrogram computation
will be done outside the model at inference time.

Usage:
    python export_onnx.py
"""

import json
import torch
import torch.nn as nn
import torchaudio
from pathlib import Path

# Import configuration
from config import SAMPLE_RATE, MODEL_NAME, OUTPUT_DIR


class SimpleWakeWordModelWithMel(nn.Module):
    """Original model with mel spectrogram inside (for loading weights)."""

    def __init__(self):
        super().__init__()

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=512,
            hop_length=160,
            n_mels=96
        )

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(128 * 12 * 18, 256)
        self.fc2 = nn.Linear(256, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.mel(x)
        x = x.unsqueeze(1)

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)

        x = self.dropout(self.relu(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))

        return x


class WakeWordClassifier(nn.Module):
    """CNN classifier only (mel spectrogram computed externally)."""

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(128 * 12 * 18, 256)
        self.fc2 = nn.Linear(256, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, 1, n_mels, time) - mel spectrogram input
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)

        x = self.dropout(self.relu(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))

        return x


if __name__ == "__main__":
    print("Loading trained model...")
    device = torch.device("cpu")

    # Load the full model
    full_model = SimpleWakeWordModelWithMel().to(device)
    full_model.load_state_dict(torch.load(OUTPUT_DIR / f"{MODEL_NAME}.pt", map_location=device))
    full_model.eval()

    # Create classifier-only model and copy weights
    classifier = WakeWordClassifier().to(device)
    classifier.conv1.load_state_dict(full_model.conv1.state_dict())
    classifier.conv2.load_state_dict(full_model.conv2.state_dict())
    classifier.conv3.load_state_dict(full_model.conv3.state_dict())
    classifier.fc1.load_state_dict(full_model.fc1.state_dict())
    classifier.fc2.load_state_dict(full_model.fc2.state_dict())
    classifier.eval()

    # Verify the classifier produces same output
    print("Verifying classifier output matches full model...")
    test_audio = torch.randn(1, int(SAMPLE_RATE * 1.5))
    with torch.no_grad():
        full_output = full_model(test_audio)
        mel_input = full_model.mel(test_audio).unsqueeze(1)
        classifier_output = classifier(mel_input)

    print(f"Full model output: {full_output.item():.6f}")
    print(f"Classifier output: {classifier_output.item():.6f}")
    assert torch.allclose(full_output, classifier_output, atol=1e-5), "Outputs don't match!"
    print("Outputs match!")

    # Export classifier to ONNX
    print("\nExporting classifier to ONNX format...")

    # Input is mel spectrogram: (batch, 1, n_mels=96, time_frames=~150)
    # With 1.5s audio at 16kHz, hop_length=160: ceil(24000/160) + 1 = 151 frames
    dummy_mel = torch.randn(1, 1, 96, 151).to(device)
    onnx_path = OUTPUT_DIR / f"{MODEL_NAME}.onnx"

    torch.onnx.export(
        classifier,
        dummy_mel,
        onnx_path,
        export_params=True,
        opset_version=17,
        input_names=['mel_spectrogram'],
        output_names=['confidence'],
        dynamic_axes={
            'mel_spectrogram': {0: 'batch_size', 3: 'time_frames'},
            'confidence': {0: 'batch_size'}
        }
    )

    print(f"ONNX model exported to: {onnx_path}")

    # Also save the mel spectrogram parameters for use at inference time
    mel_config = {
        "sample_rate": SAMPLE_RATE,
        "n_fft": 512,
        "hop_length": 160,
        "n_mels": 96,
        "audio_length_seconds": 1.5
    }

    config_path = OUTPUT_DIR / f"{MODEL_NAME}_mel_config.json"
    with open(config_path, 'w') as f:
        json.dump(mel_config, f, indent=2)
    print(f"Mel config saved to: {config_path}")

    print("\nDone!")
    print("\nNote: At inference time, compute mel spectrogram externally using these parameters:")
    print(f"  sample_rate: {SAMPLE_RATE}")
    print(f"  n_fft: 512")
    print(f"  hop_length: 160")
    print(f"  n_mels: 96")
