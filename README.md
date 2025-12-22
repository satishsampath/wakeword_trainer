# Wake Word Training Toolkit

Train custom wake word detection models using text-to-speech generated audio. This toolkit supports generating training data from multiple TTS providers (OpenAI, ElevenLabs) and training models compatible with [OpenWakeWord](https://github.com/dscripka/openWakeWord).

## Features

- **Simple GUI**: Easy-to-use desktop application for the complete workflow
- **Multi-provider TTS support**: Generate diverse training samples using OpenAI and ElevenLabs APIs
- **Negative sample generation**: Automatically create confusable words and common phrases to reduce false positives
- **Advanced data augmentation**: Noise, reverb, speed, pitch, and volume variations
- **High accuracy training**: Bidirectional GRU with focal loss achieves 98%+ accuracy
- **ONNX export**: Deploy models anywhere with ONNX runtime

## Requirements

- **Python 3.10 - 3.12** (3.13+ has compatibility issues)
- **ffmpeg** (for MP3 to WAV conversion)
- ~2GB disk space for PyTorch and dependencies

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/yourusername/wakeword_trainer.git
cd wakeword_trainer

# Run the automated setup script
python setup.py
```

The setup script will:
- Install all Python dependencies
- Download required OpenWakeWord base models
- Verify the installation

### 2. Install ffmpeg (if not already installed)

```bash
# Windows
winget install ffmpeg

# macOS
brew install ffmpeg

# Linux (Ubuntu/Debian)
sudo apt install ffmpeg
```

### 3. Run the Application

```bash
python app.py
```

Or on Windows, double-click `run.bat`

## Using the GUI

1. **Configure Wake Word**
   - Enter your wake word (e.g., "Hey Nevyx")
   - Set a model name (lowercase, underscores)

2. **Generate Training Audio**
   - Enter your OpenAI and/or ElevenLabs API keys
   - Click "Generate Positive Audio" to create wake word samples
   - Click "Generate Negative Samples" to create:
     - Confusable words (similar-sounding phrases)
     - Common phrases (everyday speech)
     - Silence/background noise

3. **Train Model**
   - Ensure "Apply data augmentation" is checked
   - Select "OpenWakeWord (recommended)" training type
   - Click "Train Model"
   - Training typically takes 2-5 minutes

4. **Export Model**
   - Click "Export to ONNX" to create the deployable model
   - Find your model in the `output/` directory

## Data Organization

```
data/
├── positive/       # Wake word audio samples
├── negative/       # General speech and silence
├── confusable/     # Similar-sounding words (hard negatives)
└── background_noise/  # (Optional) Background noise for augmentation
```

**Audio requirements:**
- Format: WAV (16-bit PCM)
- Sample rate: 16kHz (auto-resampled if different)
- Channels: Mono (auto-converted if stereo)
- Duration: ~1-2 seconds per sample

## Command Line Usage

For scripted or headless operation:

```bash
# Edit config.py with your settings
python train_openwakeword.py

# Or use the negative sample generator
python generate_negatives.py --wake-word "Hey Nevyx" --openai-key YOUR_KEY
```

## Training Tips

### For Best Results

- **40+ positive samples** with varied voices and speeds
- **80+ negative samples** including confusable words
- **Enable augmentation** to multiply your effective sample count
- **Watch the metrics**: Aim for F1 > 0.95 and FPR < 0.05

### Training Output

The trainer shows real-time metrics:
- **Loss**: Should decrease over time
- **Acc**: Training/validation accuracy
- **F1**: Harmonic mean of precision and recall (higher is better)
- **FPR**: False positive rate (lower is better)

Training stops automatically when performance plateaus.

## Model Deployment

### Python (ONNX Runtime)

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("output/hey_nevyx.onnx")

# Input: embeddings [1, 16, 96] from OpenWakeWord pipeline
embeddings = np.random.randn(1, 16, 96).astype(np.float32)
confidence = session.run(None, {"input": embeddings})[0]

if confidence > 0.5:
    print("Wake word detected!")
```

### Integration with OpenWakeWord

```python
from openwakeword.model import Model

model = Model(
    wakeword_models=["output/hey_nevyx.onnx"],
    inference_framework="onnx"
)

prediction = model.predict(audio_frame)
```

## Troubleshooting

### "Python packages not found" after restart

Make sure you're using the same Python that has the packages installed:
```bash
# Check Python version
python --version

# Use explicit path if needed
python app.py  # NOT: py app.py (on Windows)
```

### "OpenWakeWord models not found"

Run the setup script to download them:
```bash
python setup.py
```

Or manually download from [OpenWakeWord releases](https://github.com/dscripka/openWakeWord/releases) and place in `~/.openwakeword/models/`

### MP3 conversion errors

Ensure ffmpeg is installed and in your PATH:
```bash
ffmpeg -version
```

### Low accuracy / High false positives

- Add more confusable word samples
- Include more diverse negative samples
- Ensure positive samples are clean and clear
- Try recording real voice samples instead of just TTS

### TorchCodec errors

This toolkit uses `soundfile` instead of `torchaudio.load()` to avoid TorchCodec issues. If you see these errors, ensure you're using the latest version of the code.

## Project Structure

```
wakeword_trainer/
├── app.py                 # GUI application
├── setup.py               # Automated setup script
├── train_openwakeword.py  # Main training script
├── train_simple.py        # Simple CNN training (alternative)
├── generate_negatives.py  # Negative sample generator
├── augmentation.py        # Audio augmentation utilities
├── export_onnx.py         # ONNX export utility
├── requirements.txt       # Python dependencies
├── run.bat                # Windows launcher
├── models/                # Base models (created by setup)
├── data/
│   ├── positive/          # Wake word samples
│   ├── negative/          # Non-wake-word samples
│   └── confusable/        # Similar-sounding words
└── output/                # Trained models
```

## License

MIT License - feel free to use this for any project.

## Acknowledgments

- [OpenWakeWord](https://github.com/dscripka/openWakeWord) for the embedding model architecture
- OpenAI and ElevenLabs for TTS APIs
