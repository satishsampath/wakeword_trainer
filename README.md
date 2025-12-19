# Wake Word Training Toolkit

Train custom wake word detection models using text-to-speech generated audio. This toolkit supports generating training data from multiple TTS providers (OpenAI, ElevenLabs) and training models compatible with [OpenWakeWord](https://github.com/dscripka/openWakeWord).

## Features

- **Simple GUI**: Easy-to-use desktop application for the complete workflow
- **Multi-provider TTS support**: Generate diverse training samples using OpenAI and ElevenLabs APIs
- **Two training approaches**:
  - Simple CNN model (standalone, no dependencies)
  - OpenWakeWord-compatible model (works with existing OWW pipelines)
- **Data augmentation**: Built-in support for noise, reverb, time stretching, and volume variations
- **ONNX export**: Deploy models anywhere with ONNX runtime

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
python app.py
```

This opens a GUI where you can:
1. Enter your wake word (e.g., "Hey Jarvis")
2. Enter your OpenAI and/or ElevenLabs API keys
3. Click "Generate TTS Audio" to create training samples
4. Click "Train Model" to train your wake word model

![Wake Word Trainer GUI](screenshot.png)

### Alternative: Command Line

If you prefer the command line, you can edit `config.py` directly and run the training scripts:

```python
# config.py
WAKE_WORD = "Hey Jarvis"  # Your wake word phrase
MODEL_NAME = "hey_jarvis"  # Output model name (lowercase, underscores)
```

Then run:
```bash
python train_simple.py        # Simple CNN model
# or
python train_openwakeword.py  # OpenWakeWord-compatible model
```

## Data Organization

Place your audio files in the following structure:

```
data/
├── positive/     # WAV files of your wake word being spoken
│   ├── sample1.wav
│   ├── sample2.wav
│   └── ...
└── negative/     # WAV files of other speech (non-wake-word)
    ├── noise1.wav
    ├── other_speech.wav
    └── ...
```

**Audio requirements:**
- Format: WAV (16-bit PCM recommended)
- Sample rate: 16kHz (will be resampled automatically if different)
- Channels: Mono (will be converted automatically if stereo)
- Duration: ~1-2 seconds per sample

### 5. Train Your Model

**Option A: Simple CNN Model** (no external dependencies)

```bash
python train_simple.py
```

**Option B: OpenWakeWord-Compatible Model** (requires OWW embedding models)

```bash
python train_openwakeword.py
```

### 6. Find Your Model

Trained models are saved to the `output/` directory:

- `{model_name}.onnx` - ONNX model for deployment
- `{model_name}.pt` - PyTorch weights
- `{model_name}_info.json` - Model metadata

## Generating Training Data

### Using OpenAI TTS

```python
from openai import OpenAI

client = OpenAI()
voices = ["alloy", "echo", "fable", "nova", "onyx", "shimmer"]
speeds = [0.9, 0.95, 1.0, 1.05, 1.1]

for voice in voices:
    for speed in speeds:
        response = client.audio.speech.create(
            model="tts-1-hd",
            voice=voice,
            input="Hey Jarvis",
            speed=speed
        )
        response.stream_to_file(f"data/positive/{voice}_{speed}.mp3")
```

### Using ElevenLabs TTS

```python
from elevenlabs import generate, save

voices = ["voice_id_1", "voice_id_2", ...]  # Get voice IDs from ElevenLabs

for voice_id in voices:
    audio = generate(
        text="Hey Jarvis",
        voice=voice_id,
        model="eleven_multilingual_v2"
    )
    save(audio, f"data/positive/elevenlabs_{voice_id}.mp3")
```

### Converting MP3 to WAV

Use ffmpeg to convert MP3 files to the required WAV format:

```bash
for f in data/positive/*.mp3; do
    ffmpeg -i "$f" -ar 16000 -ac 1 "${f%.mp3}.wav"
done
```

## Data Augmentation

For better model robustness, augment your training data with variations:

```python
import torchaudio
import torch

def add_noise(waveform, noise_level=0.005):
    noise = torch.randn_like(waveform) * noise_level
    return waveform + noise

def change_speed(waveform, sr, speed_factor):
    effects = [["tempo", str(speed_factor)]]
    augmented, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, sr, effects)
    return augmented

def change_volume(waveform, gain_db):
    gain = 10 ** (gain_db / 20)
    return waveform * gain
```

## Training Options

### Simple CNN (`train_simple.py`)

- **Pros**: No external dependencies, works standalone
- **Cons**: Less accurate, larger model size
- **Best for**: Quick prototypes, embedded systems

### OpenWakeWord-Compatible (`train_openwakeword.py`)

- **Pros**: High accuracy, small model size (~200KB), works with OWW ecosystem
- **Cons**: Requires OpenWakeWord embedding models
- **Best for**: Production deployments, integration with existing OWW systems

To use the OpenWakeWord training, you need the embedding models:

1. Download from [OpenWakeWord releases](https://github.com/dscripka/openWakeWord/releases)
2. Or install OpenWakeWord: `pip install openwakeword`
3. Update `config.py` with the path to `melspectrogram.onnx` and `embedding_model.onnx`

## Model Deployment

### Python (ONNX Runtime)

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("output/hey_jarvis.onnx")

# For OpenWakeWord-compatible model:
# Input: embeddings [1, 16, 96]
embeddings = np.random.randn(1, 16, 96).astype(np.float32)
confidence = session.run(None, {"input": embeddings})[0]

# For simple CNN model:
# Input: audio samples [1, 24000] (1.5 seconds at 16kHz)
audio = np.random.randn(1, 24000).astype(np.float32)
confidence = session.run(None, {"audio": audio})[0]
```

### Integration with OpenWakeWord

```python
from openwakeword.model import Model

# Load your custom model alongside built-in models
model = Model(
    wakeword_models=["output/hey_jarvis.onnx"],
    inference_framework="onnx"
)

# Use with audio stream
prediction = model.predict(audio_frame)
```

## Configuration Reference

All settings are in `config.py`:

| Setting | Description | Default |
|---------|-------------|---------|
| `WAKE_WORD` | The phrase to detect | "Hey Assistant" |
| `MODEL_NAME` | Output filename prefix | "hey_assistant" |
| `DATA_DIR` | Training data location | `./data` |
| `OUTPUT_DIR` | Model output location | `./output` |
| `SAMPLE_RATE` | Audio sample rate | 16000 |
| `BATCH_SIZE` | Training batch size | 32 |
| `EPOCHS` | Max training epochs | 100 |
| `LEARNING_RATE` | Optimizer learning rate | 0.001 |

## Troubleshooting

### "No positive samples found"

Ensure your WAV files are in `data/positive/` and have the `.wav` extension.

### "OpenWakeWord models not found"

Download the embedding models or update `OPENWAKEWORD_MODELS_DIR` in `config.py`.

### Low accuracy

- Add more diverse training samples
- Include more negative samples
- Try data augmentation
- Increase training epochs

### Model too large

Use `train_openwakeword.py` for smaller models (~200KB vs ~2MB).

## Project Structure

```
wakeword_training/
├── app.py                 # GUI application (run this!)
├── config.py              # Configuration settings
├── train_simple.py        # Simple CNN training script
├── train_openwakeword.py  # OpenWakeWord-compatible training
├── export_onnx.py         # ONNX export utility
├── debug_embeddings.py    # Embedding debugging tool
├── requirements.txt       # Python dependencies
├── data/
│   ├── positive/          # Wake word audio samples
│   └── negative/          # Non-wake-word audio samples
└── output/                # Trained models
```

## License

MIT License - feel free to use this for any project.

## Acknowledgments

- [OpenWakeWord](https://github.com/dscripka/openWakeWord) for the embedding model architecture
- OpenAI and ElevenLabs for TTS APIs
