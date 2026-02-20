#!/usr/bin/env python3
"""
Scan an audio file for false positive wake word detections.

Replicates the exact same 3-model ONNX pipeline as tackl's OpenWakeWordDetector:
  melspectrogram.onnx -> embedding_model.onnx -> listen_up_tackle.onnx (streaming GRU)

With --save, saves 16-second WAV clips of each detection to data/confusable/
for use as hard negatives during retraining.

Usage:
    python verify_false_positives.py path/to/file.mp3          # count only
    python verify_false_positives.py path/to/file.mp3 --save   # count + save clips
"""

import sys
import struct
import subprocess
import math
import numpy as np
import onnxruntime as ort
from pathlib import Path

# Pipeline constants (must match OpenWakeWordDetector.kt)
SAMPLE_RATE = 16000
FRAME_SIZE = 1280  # 80ms of audio at 16kHz
MEL_BINS = 32
MEL_FRAMES_FOR_EMBEDDING = 76
MEL_STEP_SIZE = 8
EMBEDDING_DIM = 96
GRU_HIDDEN_DIM = 64
GRU_NUM_LAYERS = 2
DETECTION_THRESHOLD = 0.8
TRIGGER_LEVEL = 3
DETECTION_COOLDOWN_MS = 2000
MAX_MEL_BUFFER_SIZE = 970

# Circular buffer: 16 seconds at 16kHz (matches Kotlin FalsePositiveCollector)
CIRCULAR_BUFFER_SAMPLES = 256000

# Model paths (relative to this script's directory)
SCRIPT_DIR = Path(__file__).parent
MEL_MODEL_PATH = SCRIPT_DIR / "models" / "melspectrogram.onnx"
EMBEDDING_MODEL_PATH = SCRIPT_DIR / "models" / "embedding_model.onnx"
WAKE_WORD_MODEL_PATH = SCRIPT_DIR / "output" / "listen_up_tackle.onnx"
CONFUSABLE_DIR = SCRIPT_DIR / "data" / "confusable"


def save_circular_buffer_as_wav(circular_buffer, write_pos, filled, output_path):
    """Save the circular buffer contents as a 16kHz mono 16-bit WAV file."""
    if filled:
        sample_count = len(circular_buffer)
        linear = np.concatenate([circular_buffer[write_pos:], circular_buffer[:write_pos]])
    else:
        sample_count = write_pos
        linear = circular_buffer[:write_pos]

    if sample_count == 0:
        return

    pcm_bytes = linear.astype(np.int16).tobytes()
    data_size = len(pcm_bytes)

    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36 + data_size, b'WAVE',
        b'fmt ', 16, 1, 1, SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16,
        b'data', data_size,
    )
    with open(output_path, 'wb') as f:
        f.write(header)
        f.write(pcm_bytes)


def decode_mp3_to_pcm(audio_path):
    """Decode audio file to raw 16kHz mono s16le PCM via ffmpeg."""
    ffmpeg_candidates = ["C:/ffmpeg/bin/ffmpeg.exe", "ffmpeg"]
    ffmpeg = None
    for candidate in ffmpeg_candidates:
        try:
            subprocess.run([candidate, "-version"], capture_output=True, check=True)
            ffmpeg = candidate
            break
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue

    if ffmpeg is None:
        print("ERROR: ffmpeg not found")
        sys.exit(1)

    process = subprocess.Popen(
        [ffmpeg, "-i", str(audio_path),
         "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1",
         "-loglevel", "quiet",
         "pipe:1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return process


def compute_mel_spectrogram(mel_session, mel_input_name, audio_frame):
    """
    Compute mel spectrogram from a 1280-sample audio frame.
    Returns list of mel frames (each with MEL_BINS values), with transform applied.
    """
    input_data = audio_frame.reshape(1, -1).astype(np.float32)
    outputs = mel_session.run(None, {mel_input_name: input_data})
    output = outputs[0]

    # Squeeze batch/channel dims to get [num_frames, mel_bins]
    if output.ndim == 4:
        output = output[0, 0]
    elif output.ndim == 3:
        output = output[0]
    elif output.ndim == 1:
        output = output.reshape(1, -1)

    # Apply same transform as Kotlin: (x / 10.0) + 2.0
    output = (output / 10.0) + 2.0

    return [output[i] for i in range(output.shape[0])]


def compute_embedding(emb_session, emb_input_name, mel_window):
    """
    Compute embedding from 76 mel frames.
    Input shape: [1, 76, 32, 1] -> output flattened to 96-dim.
    """
    mel_array = np.array(mel_window, dtype=np.float32)
    input_data = mel_array.reshape(1, MEL_FRAMES_FOR_EMBEDDING, MEL_BINS, 1)
    outputs = emb_session.run(None, {emb_input_name: input_data})
    embedding = outputs[0].flatten()[:EMBEDDING_DIM]
    return embedding


def detect_wake_word_streaming(wake_session, wake_input_names, embedding, gru_hidden_state):
    """
    Feed one embedding to the streaming wake word model.
    Returns (score, updated_hidden_state).
    """
    # Embedding tensor: [1, 1, EMBEDDING_DIM]
    emb_data = np.zeros((1, 1, EMBEDDING_DIM), dtype=np.float32)
    copy_len = min(len(embedding), EMBEDDING_DIM)
    emb_data[0, 0, :copy_len] = embedding[:copy_len]

    # Hidden state tensor: [NUM_LAYERS, 1, HIDDEN_DIM]
    hidden_input = gru_hidden_state.copy()

    inputs = {
        wake_input_names[0]: emb_data,
        wake_input_names[1]: hidden_input,
    }
    outputs = wake_session.run(None, inputs)

    # Output 0: score
    raw_score = float(outputs[0].flatten()[0])

    # Output 1: new hidden state
    new_hidden = outputs[1].astype(np.float32)

    # Apply sigmoid if raw score is outside [0, 1] (same as Kotlin)
    if raw_score < 0.0 or raw_score > 1.0:
        score = 1.0 / (1.0 + math.exp(-raw_score))
    else:
        score = raw_score

    return score, new_hidden


def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_false_positives.py <audio_file> [--save]")
        sys.exit(1)

    audio_path = Path(sys.argv[1])
    save_clips = "--save" in sys.argv

    if not audio_path.exists():
        print(f"ERROR: File not found: {audio_path}")
        sys.exit(1)

    # Load ONNX models
    print(f"Loading models...")
    mel_session = ort.InferenceSession(str(MEL_MODEL_PATH))
    mel_input_name = mel_session.get_inputs()[0].name

    emb_session = ort.InferenceSession(str(EMBEDDING_MODEL_PATH))
    emb_input_name = emb_session.get_inputs()[0].name

    wake_session = ort.InferenceSession(str(WAKE_WORD_MODEL_PATH))
    wake_input_names = [inp.name for inp in wake_session.get_inputs()]

    print(f"Models loaded. Scanning: {audio_path.name}")
    if save_clips:
        CONFUSABLE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Saving clips to: {CONFUSABLE_DIR}")
    print()

    # Decode audio
    process = decode_mp3_to_pcm(audio_path)
    pcm_stream = process.stdout

    # Pipeline state
    mel_buffer = []
    gru_hidden_state = np.zeros((GRU_NUM_LAYERS, 1, GRU_HIDDEN_DIM), dtype=np.float32)
    consecutive_activations = 0
    detection_count = 0
    last_detection_frame = 0
    frames_processed = 0

    # Circular buffer for saving clips (16 seconds of audio)
    circular_buffer = np.zeros(CIRCULAR_BUFFER_SAMPLES, dtype=np.int16)
    buffer_pos = 0
    buffer_filled = False

    # Cooldown in frames (audio time, not wall-clock)
    cooldown_frames = DETECTION_COOLDOWN_MS * SAMPLE_RATE // (1000 * FRAME_SIZE)

    bytes_per_frame = FRAME_SIZE * 2  # 16-bit samples = 2 bytes each

    while True:
        # Read one frame of raw PCM (1280 samples = 2560 bytes)
        raw_bytes = b""
        while len(raw_bytes) < bytes_per_frame:
            chunk = pcm_stream.read(bytes_per_frame - len(raw_bytes))
            if not chunk:
                break
            raw_bytes += chunk
        if len(raw_bytes) < bytes_per_frame:
            break  # EOF

        # Convert bytes to shorts (little-endian) then normalize to float
        shorts = np.frombuffer(raw_bytes, dtype='<i2')
        float_frame = shorts.astype(np.float32) / 32768.0

        # Feed into circular buffer
        if save_clips:
            for s in shorts:
                circular_buffer[buffer_pos] = s
                buffer_pos += 1
                if buffer_pos >= CIRCULAR_BUFFER_SAMPLES:
                    buffer_pos = 0
                    buffer_filled = True

        # Step 1: Mel spectrogram
        mel_frames = compute_mel_spectrogram(mel_session, mel_input_name, float_frame)
        if not mel_frames:
            frames_processed += 1
            continue

        mel_buffer.extend(mel_frames)

        # Trim buffer if too large
        while len(mel_buffer) > MAX_MEL_BUFFER_SIZE:
            mel_buffer.pop(0)

        # Step 2 & 3: Embedding + streaming detection
        last_score = 0.0
        embedding_count = 0

        while len(mel_buffer) >= MEL_FRAMES_FOR_EMBEDDING:
            # Extract window of 76 frames
            window = mel_buffer[:MEL_FRAMES_FOR_EMBEDDING]
            embedding = compute_embedding(emb_session, emb_input_name, window)

            embedding_count += 1

            # Feed to streaming model
            last_score, gru_hidden_state = detect_wake_word_streaming(
                wake_session, wake_input_names, embedding, gru_hidden_state
            )

            # Slide mel buffer by step size
            for _ in range(MEL_STEP_SIZE):
                if mel_buffer:
                    mel_buffer.pop(0)

        if embedding_count == 0:
            last_score = 0.0

        frames_processed += 1

        # Detection logic (same as Kotlin handleDetectionScore)
        if last_score == 0.0:
            continue

        if last_score > DETECTION_THRESHOLD:
            consecutive_activations += 1
            if consecutive_activations >= TRIGGER_LEVEL:
                if frames_processed - last_detection_frame > cooldown_frames:
                    last_detection_frame = frames_processed
                    detection_count += 1

                    elapsed_sec = frames_processed * FRAME_SIZE // SAMPLE_RATE
                    minutes = elapsed_sec // 60
                    seconds = elapsed_sec % 60

                    if save_clips:
                        filename = f"false_positive_{detection_count:03d}.wav"
                        save_circular_buffer_as_wav(
                            circular_buffer, buffer_pos, buffer_filled,
                            CONFUSABLE_DIR / filename,
                        )
                        print(f"  [{detection_count}] score={last_score:.4f}  at {minutes}:{seconds:02d}  saved: {filename}")
                    else:
                        print(f"  [{detection_count}] score={last_score:.4f}  at {minutes}:{seconds:02d}")

                consecutive_activations = 0
                gru_hidden_state = np.zeros(
                    (GRU_NUM_LAYERS, 1, GRU_HIDDEN_DIM), dtype=np.float32
                )
        else:
            consecutive_activations = 0

    # Cleanup
    pcm_stream.close()
    process.wait()

    total_sec = frames_processed * FRAME_SIZE // SAMPLE_RATE
    print()
    print(f"Done! {detection_count} false positives in {total_sec // 60}m {total_sec % 60}s of audio.")


if __name__ == "__main__":
    main()
