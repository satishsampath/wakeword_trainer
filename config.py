#!/usr/bin/env python3
"""
Configuration for Wake Word Training

Edit this file to configure your wake word training settings.
"""

import os
from pathlib import Path

# =============================================================================
# WAKE WORD CONFIGURATION
# =============================================================================

# The wake word phrase you want to train (e.g., "Hey Jarvis", "OK Computer")
WAKE_WORD = "Hey Nevyx"

# Model name (used for output filenames, should be lowercase with underscores)
MODEL_NAME = "hey_nevyx"

# =============================================================================
# DIRECTORY CONFIGURATION
# =============================================================================

# Base directory for your wake word data
# This directory should contain 'positive/' and 'negative/' subdirectories
DATA_DIR = Path(__file__).parent / "data"

# Output directory for trained models
OUTPUT_DIR = Path(__file__).parent / "output"

# =============================================================================
# OPENWAKEWORD MODEL PATHS (for embedding-based training)
# =============================================================================

# These models are required for train_openwakeword.py
# The setup.py script will download them automatically.
#
# Search order:
# 1. Local 'models/' directory (created by setup.py)
# 2. User's ~/.openwakeword/models/ directory

def _find_model(name):
    """Find a model file in known locations."""
    locations = [
        Path(__file__).parent / "models" / name,  # Local models/ dir
        Path(os.path.expanduser("~")) / ".openwakeword" / "models" / name,  # User dir
    ]
    for path in locations:
        if path.exists():
            return path
    # Return default path (will error later if not found)
    return locations[0]

MEL_MODEL_PATH = _find_model("melspectrogram.onnx")
EMBEDDING_MODEL_PATH = _find_model("embedding_model.onnx")

# =============================================================================
# AUDIO PARAMETERS
# =============================================================================

SAMPLE_RATE = 16000  # 16kHz is standard for wake word detection

# =============================================================================
# TRAINING PARAMETERS
# =============================================================================

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001

# =============================================================================
# MODEL ARCHITECTURE PARAMETERS (for OpenWakeWord-compatible training)
# =============================================================================

SAMPLES_PER_FRAME = 1280  # 80ms at 16kHz
MEL_BANDS = 32
EMBEDDING_SIZE = 96
FEATURE_FRAMES = 16  # Number of embedding frames for classification
