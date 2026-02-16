#!/usr/bin/env python3
"""
Train a streaming wake word model.

Unlike the snapshot approach (feed 16 embeddings, get 1 score), this streaming
model processes one embedding at a time with persistent GRU hidden state.
This allows it to learn temporal ordering — "listen up" followed by "tackle"
produces a different hidden state trajectory than "tackle" alone.

Pipeline:
    Raw Audio -> Mel Spectrogram -> Embedding Model -> [Streaming Classifier] -> Score

The classifier:
- Input:  embedding [1, 1, 96] + hidden state [num_layers, 1, hidden_dim]
- Output: score [1, 1] + new hidden state [num_layers, 1, hidden_dim]

Training uses sliding-window simulation: audio is padded with silence before/after,
run through the full mel+embedding pipeline, then the resulting embedding sequence
is fed to the model one step at a time. Per-timestep labels mark when the wake
phrase has fully entered the embedding window.

Usage:
    python train_openwakeword.py
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
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

# Import augmentation
try:
    from augmentation import AudioAugmenter
    HAS_AUGMENTATION = True
except ImportError:
    HAS_AUGMENTATION = False

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    print("Warning: onnxruntime not installed.")

try:
    import torchaudio
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False

# Streaming model hyperparameters
HIDDEN_DIM = 64
NUM_LAYERS = 2
SILENCE_PAD_SECONDS = 2.0  # Silence before and after utterance


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
        """Extract embeddings from audio. Returns [num_embeddings, 96]."""
        mel_buffer = []
        for i in range(0, len(audio_samples) - SAMPLES_PER_FRAME + 1, SAMPLES_PER_FRAME):
            chunk = audio_samples[i:i + SAMPLES_PER_FRAME]
            mel_output = self._compute_mel(chunk)
            for j in range(mel_output.shape[0]):
                mel_buffer.append(mel_output[j])

        STRIDE = 8
        embeddings = []
        for window_start in range(0, len(mel_buffer) - 76 + 1, STRIDE):
            mel_window = np.array(mel_buffer[window_start:window_start + 76])
            embedding = self._compute_embedding(mel_window)
            embeddings.append(embedding)

        return np.array(embeddings) if embeddings else np.zeros((0, EMBEDDING_SIZE))

    def extract_embeddings_with_timing(self, audio_samples):
        """
        Extract embeddings AND track which audio sample offset each embedding covers.
        Returns (embeddings, embedding_audio_ranges) where each range is
        (start_sample, end_sample) in the original audio.
        """
        mel_buffer = []
        # Each audio frame produces multiple mel frames. Track the mapping.
        mel_frame_audio_offset = []  # audio sample offset for each mel frame

        for i in range(0, len(audio_samples) - SAMPLES_PER_FRAME + 1, SAMPLES_PER_FRAME):
            chunk = audio_samples[i:i + SAMPLES_PER_FRAME]
            mel_output = self._compute_mel(chunk)
            for j in range(mel_output.shape[0]):
                mel_buffer.append(mel_output[j])
                mel_frame_audio_offset.append(i)

        STRIDE = 8
        embeddings = []
        emb_ranges = []
        for window_start in range(0, len(mel_buffer) - 76 + 1, STRIDE):
            mel_window = np.array(mel_buffer[window_start:window_start + 76])
            embedding = self._compute_embedding(mel_window)
            embeddings.append(embedding)
            # This embedding covers mel frames [window_start, window_start+76)
            audio_start = mel_frame_audio_offset[window_start]
            audio_end = mel_frame_audio_offset[min(window_start + 75, len(mel_frame_audio_offset) - 1)] + SAMPLES_PER_FRAME
            emb_ranges.append((audio_start, audio_end))

        if embeddings:
            return np.array(embeddings), emb_ranges
        else:
            return np.zeros((0, EMBEDDING_SIZE)), []

    def _compute_mel(self, samples):
        input_data = samples.reshape(1, -1).astype(np.float32)
        outputs = self.mel_session.run(None, {self.mel_input_name: input_data})
        output = outputs[0]
        if output.ndim == 4:
            output = output[0, 0]
        elif output.ndim == 3:
            output = output[0]
        output = (output / 10.0) + 2.0
        return output

    def _compute_embedding(self, mel_frames):
        input_data = mel_frames.reshape(1, 76, MEL_BANDS, 1).astype(np.float32)
        outputs = self.embedding_session.run(None, {self.embedding_input_name: input_data})
        return outputs[0].flatten()[:EMBEDDING_SIZE]


class StreamingWakeWordDataset(Dataset):
    """
    Dataset for streaming wake word training.

    Each sample is a variable-length sequence of embeddings with per-timestep labels.
    Audio is padded with silence before/after, run through the full pipeline,
    and labeled based on when the wake phrase has fully entered the embedding window.
    """

    def __init__(self, positive_files, negative_files, confusable_files,
                 embedder, augmenter=None, augment_factor=3):
        self.sequences = []   # List of (embeddings_array, labels_array)
        self.embedder = embedder
        self.augmenter = augmenter
        self.augment_factor = augment_factor

        # Process positive samples
        print(f"\nProcessing {len(positive_files)} positive files...")
        for i, wav_file in enumerate(positive_files):
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(positive_files)}...")
            self._process_positive(wav_file)

        # Process confusable samples (hard negatives)
        if confusable_files:
            print(f"\nProcessing {len(confusable_files)} confusable files...")
            for i, wav_file in enumerate(confusable_files):
                if (i + 1) % 100 == 0:
                    print(f"  {i+1}/{len(confusable_files)}...")
                self._process_negative(wav_file)

        # Process general negatives
        print(f"\nProcessing {len(negative_files)} negative files...")
        for i, wav_file in enumerate(negative_files):
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(negative_files)}...")
            self._process_negative(wav_file)

        pos_steps = sum(l.sum().item() for _, l in self.sequences)
        neg_steps = sum((1 - l).sum().item() for _, l in self.sequences)
        print(f"\nDataset: {len(self.sequences)} sequences, "
              f"{int(pos_steps)} positive steps, {int(neg_steps)} negative steps")

    def _load_audio(self, wav_path):
        try:
            import soundfile as sf
            from scipy import signal
            audio, sr = sf.read(wav_path, dtype='float32')
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            if sr != SAMPLE_RATE:
                num_samples = int(len(audio) * SAMPLE_RATE / sr)
                audio = signal.resample(audio, num_samples)
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val * 0.9
            return audio
        except Exception:
            return None

    def _pad_with_silence(self, audio, pad_seconds=SILENCE_PAD_SECONDS):
        """Pad audio with silence before and after."""
        pad_samples = int(SAMPLE_RATE * pad_seconds)
        return np.concatenate([
            np.zeros(pad_samples, dtype=np.float32),
            audio,
            np.zeros(pad_samples, dtype=np.float32)
        ])

    def _process_positive(self, wav_path):
        """Process a positive sample: create sequence with per-timestep labels."""
        audio = self._load_audio(wav_path)
        if audio is None:
            return

        audio_variants = [audio]
        if self.augmenter and self.augment_factor > 1:
            for _ in range(self.augment_factor - 1):
                intensity = random.choice(['light', 'medium', 'heavy'])
                aug_audio, _ = self.augmenter.random_augment(audio, intensity)
                audio_variants.append(aug_audio)

        for audio_var in audio_variants:
            # Pad with silence to simulate real inference
            padded = self._pad_with_silence(audio_var)
            speech_start = int(SAMPLE_RATE * SILENCE_PAD_SECONDS)
            speech_end = speech_start + len(audio_var)

            # Extract embeddings with timing info
            embeddings, emb_ranges = self.embedder.extract_embeddings_with_timing(padded)
            if len(embeddings) < 4:
                continue

            # Label each embedding: 1 if the speech content is mostly within its range
            # An embedding covers ~76 mel frames. Mark as positive when the embedding's
            # audio range overlaps significantly with the speech region AND
            # the speech has been going for a while (the model needs to have "heard"
            # enough of the phrase).
            labels = np.zeros(len(embeddings), dtype=np.float32)
            for j, (a_start, a_end) in enumerate(emb_ranges):
                # How much of this embedding's audio range is speech?
                overlap_start = max(a_start, speech_start)
                overlap_end = min(a_end, speech_end)
                overlap = max(0, overlap_end - overlap_start)
                emb_duration = a_end - a_start

                # Also check: has at least 60% of the speech been heard by this point?
                speech_heard = max(0, min(a_end, speech_end) - speech_start)
                speech_total = speech_end - speech_start
                fraction_heard = speech_heard / speech_total if speech_total > 0 else 0

                # Label = 1 when: embedding overlaps with speech AND
                # we've heard most of the phrase
                if overlap > 0.3 * emb_duration and fraction_heard >= 0.6:
                    labels[j] = 1.0

            emb_tensor = torch.FloatTensor(embeddings)
            label_tensor = torch.FloatTensor(labels)
            self.sequences.append((emb_tensor, label_tensor))

    def _process_negative(self, wav_path):
        """Process a negative sample: all timestep labels are 0."""
        audio = self._load_audio(wav_path)
        if audio is None:
            return

        audio_variants = [audio]
        if self.augmenter and self.augment_factor > 1:
            for _ in range(self.augment_factor - 1):
                intensity = random.choice(['light', 'medium', 'heavy'])
                aug_audio, _ = self.augmenter.random_augment(audio, intensity)
                audio_variants.append(aug_audio)

        for audio_var in audio_variants:
            padded = self._pad_with_silence(audio_var)
            embeddings = self.embedder.extract_embeddings(padded)
            if len(embeddings) < 4:
                continue

            emb_tensor = torch.FloatTensor(embeddings)
            label_tensor = torch.zeros(len(embeddings))
            self.sequences.append((emb_tensor, label_tensor))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def collate_sequences(batch):
    """Collate variable-length sequences by padding to max length in batch."""
    embeddings = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # Pad sequences to same length
    padded_emb = pad_sequence(embeddings, batch_first=True)  # [B, max_len, 96]
    padded_labels = pad_sequence(labels, batch_first=True)    # [B, max_len]

    # Create mask (1 for real data, 0 for padding)
    lengths = torch.tensor([len(e) for e in embeddings])
    max_len = padded_emb.size(1)
    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)  # [B, max_len]

    return padded_emb, padded_labels, mask


class StreamingWakeWordClassifier(nn.Module):
    """
    Streaming (causal) wake word classifier.

    Processes embeddings one at a time, maintaining GRU hidden state.
    Can also process full sequences during training.

    Input:  [batch, seq_len, 96] + optional hidden state
    Output: [batch, seq_len, 1] scores + new hidden state
    """

    def __init__(self, input_size=EMBEDDING_SIZE, hidden_size=HIDDEN_DIM, num_layers=NUM_LAYERS):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)

        # Unidirectional GRU (causal — can only see past)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )

        # Per-timestep output
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x, h=None):
        """
        Args:
            x: [batch, seq_len, input_size] or [batch, 1, input_size] for streaming
            h: [num_layers, batch, hidden_size] or None (zeros)
        Returns:
            scores: [batch, seq_len]
            h_new: [num_layers, batch, hidden_size]
        """
        batch_size = x.size(0)
        if h is None:
            h = torch.zeros(self.num_layers, batch_size, self.hidden_size,
                            device=x.device, dtype=x.dtype)

        x = self.input_proj(x)          # [batch, seq_len, hidden]
        gru_out, h_new = self.gru(x, h)  # [batch, seq_len, hidden]
        scores = self.fc(gru_out)         # [batch, seq_len, 1]
        return scores.squeeze(-1), h_new  # [batch, seq_len], [layers, batch, hidden]


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance. Focuses on hard examples."""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets, mask=None):
        bce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        if mask is not None:
            focal_loss = focal_loss * mask.float()
            return focal_loss.sum() / mask.float().sum().clamp(min=1)
        return focal_loss.mean()


def train_model():
    """Main training function."""
    print("=" * 60)
    print(f"Streaming Wake Word Training: {WAKE_WORD}")
    print("=" * 60)
    print(f"\nData directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Architecture: Streaming GRU (hidden={HIDDEN_DIM}, layers={NUM_LAYERS})")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not HAS_ONNX:
        print("\nERROR: onnxruntime required!")
        sys.exit(1)

    if not MEL_MODEL_PATH.exists() or not EMBEDDING_MODEL_PATH.exists():
        print(f"ERROR: OpenWakeWord models not found!")
        sys.exit(1)

    embedder = OpenWakeWordEmbedder(MEL_MODEL_PATH, EMBEDDING_MODEL_PATH)

    augmenter = None
    if HAS_AUGMENTATION:
        augmenter = AudioAugmenter(SAMPLE_RATE)
        noise_dir = DATA_DIR / "background_noise"
        if noise_dir.exists():
            augmenter.load_background_noises(noise_dir)

    # Get audio files
    positive_files = list((DATA_DIR / "positive").glob("*.wav"))
    negative_files = list((DATA_DIR / "negative").glob("*.wav"))
    confusable_dir = DATA_DIR / "confusable"
    confusable_files = list(confusable_dir.glob("*.wav")) if confusable_dir.exists() else []

    print(f"\nFound audio files:")
    print(f"  Positive (wake word): {len(positive_files)}")
    print(f"  Negative (general): {len(negative_files)}")
    print(f"  Confusable (hard negative): {len(confusable_files)}")

    if len(positive_files) == 0:
        print("\nERROR: No positive samples found!")
        sys.exit(1)

    # Create dataset
    print("\n" + "=" * 40)
    print("Building Streaming Dataset")
    print("=" * 40)

    total_raw = len(positive_files) + len(negative_files) + len(confusable_files)
    augment_factor = 10 if total_raw < 100 else 5 if total_raw < 500 else 3
    print(f"Augmentation factor: {augment_factor}x per sample")

    dataset = StreamingWakeWordDataset(
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

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, collate_fn=collate_sequences)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0,
                            collate_fn=collate_sequences)

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = StreamingWakeWordClassifier().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

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
        # Training
        model.train()
        train_loss = 0
        train_tp, train_fp, train_fn, train_tn = 0, 0, 0, 0
        train_batches = 0

        for batch_emb, batch_labels, batch_mask in train_loader:
            batch_emb = batch_emb.to(device)
            batch_labels = batch_labels.to(device)
            batch_mask = batch_mask.to(device)

            optimizer.zero_grad()
            scores, _ = model(batch_emb)  # [B, seq_len]
            loss = criterion(scores, batch_labels, batch_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

            # Metrics on masked positions only
            with torch.no_grad():
                pred = (scores > 0.5).long()
                valid = batch_mask
                train_tp += ((pred == 1) & (batch_labels.long() == 1) & valid).sum().item()
                train_fp += ((pred == 1) & (batch_labels.long() == 0) & valid).sum().item()
                train_fn += ((pred == 0) & (batch_labels.long() == 1) & valid).sum().item()
                train_tn += ((pred == 0) & (batch_labels.long() == 0) & valid).sum().item()

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0
        val_tp, val_fp, val_fn, val_tn = 0, 0, 0, 0
        val_batches = 0

        with torch.no_grad():
            for batch_emb, batch_labels, batch_mask in val_loader:
                batch_emb = batch_emb.to(device)
                batch_labels = batch_labels.to(device)
                batch_mask = batch_mask.to(device)

                scores, _ = model(batch_emb)
                loss = criterion(scores, batch_labels, batch_mask)
                val_loss += loss.item()
                val_batches += 1

                pred = (scores > 0.5).long()
                valid = batch_mask
                val_tp += ((pred == 1) & (batch_labels.long() == 1) & valid).sum().item()
                val_fp += ((pred == 1) & (batch_labels.long() == 0) & valid).sum().item()
                val_fn += ((pred == 0) & (batch_labels.long() == 1) & valid).sum().item()
                val_tn += ((pred == 0) & (batch_labels.long() == 0) & valid).sum().item()

        # Metrics
        train_total = train_tp + train_fp + train_fn + train_tn
        train_acc = (train_tp + train_tn) / train_total if train_total > 0 else 0
        val_total = val_tp + val_fp + val_fn + val_tn
        val_acc = (val_tp + val_tn) / val_total if val_total > 0 else 0

        val_precision = val_tp / (val_tp + val_fp) if (val_tp + val_fp) > 0 else 0
        val_recall = val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 0
        val_f1 = 2 * val_precision * val_recall / (val_precision + val_recall) if (val_precision + val_recall) > 0 else 0
        val_fpr = val_fp / (val_fp + val_tn) if (val_fp + val_tn) > 0 else 0

        avg_train_loss = train_loss / max(train_batches, 1)
        avg_val_loss = val_loss / max(val_batches, 1)

        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Loss: {avg_train_loss:.4f}/{avg_val_loss:.4f} | "
              f"Acc: {train_acc:.3f}/{val_acc:.3f} | "
              f"F1: {val_f1:.3f} | FPR: {val_fpr:.3f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), OUTPUT_DIR / f"{MODEL_NAME}_classifier.pt")
            print(f"  -> Saved best model (F1: {val_f1:.4f}, Acc: {val_acc:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= max_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print(f"\n{'=' * 40}")
    print("Training Complete!")
    print(f"{'=' * 40}")
    print(f"Best validation F1: {best_val_f1:.4f}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    # Export to ONNX with hidden state as input/output
    print("\n" + "=" * 40)
    print("Exporting to ONNX (streaming)")
    print("=" * 40)

    model.load_state_dict(torch.load(OUTPUT_DIR / f"{MODEL_NAME}_classifier.pt", map_location='cpu'))
    model = model.to('cpu')
    model.eval()

    # Streaming inference: 1 embedding at a time + hidden state
    dummy_embedding = torch.randn(1, 1, EMBEDDING_SIZE)
    dummy_hidden = torch.zeros(NUM_LAYERS, 1, HIDDEN_DIM)

    onnx_path = OUTPUT_DIR / f"{MODEL_NAME}.onnx"

    torch.onnx.export(
        model,
        (dummy_embedding, dummy_hidden),
        onnx_path,
        export_params=True,
        opset_version=11,
        input_names=['embedding', 'hidden_in'],
        output_names=['score', 'hidden_out'],
        dynamic_axes={
            'embedding': {0: 'batch'},
            'hidden_in': {1: 'batch'},
            'score': {0: 'batch'},
            'hidden_out': {1: 'batch'}
        }
    )

    print(f"ONNX model exported to: {onnx_path}")

    # Verify ONNX model
    ort_session = ort.InferenceSession(str(onnx_path))
    test_emb = np.random.randn(1, 1, EMBEDDING_SIZE).astype(np.float32)
    test_hidden = np.zeros((NUM_LAYERS, 1, HIDDEN_DIM), dtype=np.float32)

    with torch.no_grad():
        torch_score, torch_h = model(torch.FloatTensor(test_emb), torch.FloatTensor(test_hidden))

    inputs = {
        ort_session.get_inputs()[0].name: test_emb,
        ort_session.get_inputs()[1].name: test_hidden,
    }
    ort_outputs = ort_session.run(None, inputs)

    print(f"PyTorch score: {torch_score.item():.6f}")
    print(f"ONNX score:    {ort_outputs[0][0][0]:.6f}")
    print(f"Score match: {np.allclose(torch_score.numpy(), ort_outputs[0], atol=1e-4)}")
    print(f"Hidden match: {np.allclose(torch_h.numpy(), ort_outputs[1], atol=1e-4)}")

    # Also test a full sequence to verify streaming matches batch
    print("\nVerifying streaming vs batch consistency...")
    test_seq = np.random.randn(1, 10, EMBEDDING_SIZE).astype(np.float32)
    with torch.no_grad():
        batch_scores, _ = model(torch.FloatTensor(test_seq))

    # Run same sequence one step at a time
    h = np.zeros((NUM_LAYERS, 1, HIDDEN_DIM), dtype=np.float32)
    streaming_scores = []
    for t in range(10):
        step_emb = test_seq[:, t:t+1, :]
        inputs = {
            ort_session.get_inputs()[0].name: step_emb,
            ort_session.get_inputs()[1].name: h,
        }
        ort_out = ort_session.run(None, inputs)
        streaming_scores.append(ort_out[0][0][0])
        h = ort_out[1]

    batch_np = batch_scores.numpy()[0]
    stream_np = np.array(streaming_scores)
    print(f"Batch scores:     {' '.join(f'{s:.4f}' for s in batch_np)}")
    print(f"Streaming scores: {' '.join(f'{s:.4f}' for s in stream_np)}")
    print(f"Max diff: {np.max(np.abs(batch_np - stream_np)):.6f}")

    # Save model info
    model_info = {
        "name": MODEL_NAME,
        "wake_word": WAKE_WORD,
        "architecture": "streaming_gru",
        "input_embedding_shape": [1, 1, EMBEDDING_SIZE],
        "hidden_state_shape": [NUM_LAYERS, 1, HIDDEN_DIM],
        "output_score_shape": [1, 1],
        "best_f1_score": float(best_val_f1),
        "best_accuracy": float(best_val_acc),
        "training_sequences": len(dataset),
    }

    with open(OUTPUT_DIR / f"{MODEL_NAME}_info.json", 'w') as f:
        json.dump(model_info, f, indent=2)

    print(f"\n{'=' * 60}")
    print("DONE!")
    print(f"{'=' * 60}")
    print(f"\nModel: {onnx_path}")
    print(f"Size: {onnx_path.stat().st_size / 1024:.1f} KB")
    print(f"\nThis is a STREAMING model. Inference:")
    print(f"  1. Feed one embedding [1,1,96] + hidden state [{NUM_LAYERS},1,{HIDDEN_DIM}]")
    print(f"  2. Get score [1,1] + new hidden state [{NUM_LAYERS},1,{HIDDEN_DIM}]")
    print(f"  3. Pass hidden state to next call")
    print(f"  4. Reset hidden state to zeros after detection or timeout")


if __name__ == "__main__":
    train_model()
