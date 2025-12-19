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
WAKE_WORD = "Hey Assistant"

# Model name (used for output filenames, should be lowercase with underscores)
MODEL_NAME = "hey_assistant"

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

# Directory containing OpenWakeWord's melspectrogram.onnx and embedding_model.onnx
# These models are required for train_openwakeword.py
#
# You can download them from:
# https://github.com/dscripka/openWakeWord/tree/main/openwakeword/resources/models
#
# Or if you have OpenWakeWord installed via pip, find them at:
# <python-env>/lib/python3.x/site-packages/openwakeword/resources/models/
OPENWAKEWORD_MODELS_DIR = Path(os.path.expanduser("~")) / ".openwakeword" / "models"

MEL_MODEL_PATH = OPENWAKEWORD_MODELS_DIR / "melspectrogram.onnx"
EMBEDDING_MODEL_PATH = OPENWAKEWORD_MODELS_DIR / "embedding_model.onnx"

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
