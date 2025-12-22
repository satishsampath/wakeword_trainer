#!/usr/bin/env python3
"""
Debug script to check embedding extraction with OpenWakeWord models.

This script helps verify that the OpenWakeWord embedding pipeline is working
correctly with your audio files.

Usage:
    python debug_embeddings.py
"""

import os
import numpy as np
import onnxruntime as ort
import torchaudio
from pathlib import Path

# Import configuration
from config import (
    DATA_DIR, MEL_MODEL_PATH, EMBEDDING_MODEL_PATH,
    SAMPLE_RATE, SAMPLES_PER_FRAME, MEL_BANDS, EMBEDDING_SIZE
)

# Load a test audio file
positive_dir = DATA_DIR / "positive"
wav_files = list(positive_dir.glob("*.wav"))

print(f"Found {len(wav_files)} WAV files")
if len(wav_files) == 0:
    print("No WAV files found!")
    exit(1)

test_file = wav_files[0]
print(f"\nTesting with: {test_file}")

# Load audio
import soundfile as sf
from scipy import signal

audio, sr = sf.read(test_file, dtype='float32')
print(f"Original sample rate: {sr}, shape: {audio.shape}")

# Convert to mono if stereo
if len(audio.shape) > 1:
    audio = audio.mean(axis=1)
    print("Converted to mono")

# Resample if needed
if sr != SAMPLE_RATE:
    num_samples = int(len(audio) * SAMPLE_RATE / sr)
    audio = signal.resample(audio, num_samples)
    print(f"Resampled to: {SAMPLE_RATE}Hz")
print(f"Audio shape: {audio.shape}, duration: {len(audio)/SAMPLE_RATE:.2f}s")

# Pad if needed
min_samples = int(SAMPLE_RATE * 1.5)
if len(audio) < min_samples:
    audio = np.pad(audio, (0, min_samples - len(audio)))
    print(f"Padded to {len(audio)} samples")

# Load mel model
print(f"\nLoading mel model from: {MEL_MODEL_PATH}")
mel_session = ort.InferenceSession(str(MEL_MODEL_PATH))
mel_input_name = mel_session.get_inputs()[0].name
print(f"Mel input name: {mel_input_name}")
print(f"Mel input shape: {mel_session.get_inputs()[0].shape}")
print(f"Mel output shape: {mel_session.get_outputs()[0].shape}")

# Load embedding model
print(f"\nLoading embedding model from: {EMBEDDING_MODEL_PATH}")
emb_session = ort.InferenceSession(str(EMBEDDING_MODEL_PATH))
emb_input_name = emb_session.get_inputs()[0].name
print(f"Embedding input name: {emb_input_name}")
print(f"Embedding input shape: {emb_session.get_inputs()[0].shape}")
print(f"Embedding output shape: {emb_session.get_outputs()[0].shape}")

# Process one audio chunk
print(f"\n=== Processing first 1280 samples ===")
chunk = audio[:SAMPLES_PER_FRAME]
input_data = chunk.reshape(1, -1).astype(np.float32)
print(f"Mel input shape: {input_data.shape}")

mel_output = mel_session.run(None, {mel_input_name: input_data})[0]
print(f"Mel output shape: {mel_output.shape}")
print(f"Mel output range: [{mel_output.min():.4f}, {mel_output.max():.4f}]")

# Figure out mel frame structure
if mel_output.ndim == 4:
    mel_frames = mel_output[0, 0]
elif mel_output.ndim == 3:
    mel_frames = mel_output[0]
else:
    mel_frames = mel_output

print(f"Mel frames shape after reshape: {mel_frames.shape}")

# Process all audio to get mel buffer
print(f"\n=== Building mel buffer ===")
mel_buffer = []

for i in range(0, len(audio) - SAMPLES_PER_FRAME + 1, SAMPLES_PER_FRAME):
    chunk = audio[i:i + SAMPLES_PER_FRAME]
    input_data = chunk.reshape(1, -1).astype(np.float32)
    mel_output = mel_session.run(None, {mel_input_name: input_data})[0]

    if mel_output.ndim == 4:
        mel_frames = mel_output[0, 0]
    elif mel_output.ndim == 3:
        mel_frames = mel_output[0]
    else:
        mel_frames = mel_output

    # Apply normalization
    mel_frames = (mel_frames / 10.0) + 2.0

    # Add each frame to buffer
    for j in range(mel_frames.shape[0]):
        mel_buffer.append(mel_frames[j])

print(f"Total mel frames in buffer: {len(mel_buffer)}")
print(f"Each mel frame shape: {mel_buffer[0].shape if mel_buffer else 'N/A'}")

# Try to compute embeddings
print(f"\n=== Computing embeddings ===")
print(f"Need 76 mel frames for one embedding")
print(f"With stride of 8, we can get {max(0, (len(mel_buffer) - 76) // 8 + 1)} embeddings")

STRIDE = 8
embeddings = []

for window_start in range(0, len(mel_buffer) - 76 + 1, STRIDE):
    mel_window = np.array(mel_buffer[window_start:window_start + 76])
    print(f"  Window {len(embeddings)}: mel_window shape = {mel_window.shape}")

    # Input shape: [1, 76, 32, 1]
    input_data = mel_window.reshape(1, 76, MEL_BANDS, 1).astype(np.float32)
    print(f"  Embedding input shape: {input_data.shape}")

    try:
        emb_output = emb_session.run(None, {emb_input_name: input_data})[0]
        print(f"  Embedding output shape: {emb_output.shape}")
        embedding = emb_output.flatten()[:EMBEDDING_SIZE]
        embeddings.append(embedding)
        print(f"  Final embedding shape: {embedding.shape}")
    except Exception as e:
        print(f"  ERROR: {e}")
        break

    if len(embeddings) >= 5:
        print("  (stopping after 5 embeddings for debug)")
        break

print(f"\nTotal embeddings extracted: {len(embeddings)}")
print(f"Need at least 16 embeddings for one classification sample")
