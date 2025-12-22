#!/usr/bin/env python3
"""
Setup script for Wake Word Trainer

This script:
1. Checks Python version compatibility
2. Installs required packages
3. Downloads OpenWakeWord base models
4. Verifies the installation
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path


def check_python_version():
    """Ensure Python 3.10-3.12 is being used."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major != 3 or version.minor < 10 or version.minor > 12:
        print("\nWARNING: This project works best with Python 3.10-3.12")
        print("  Python 3.13+ may have compatibility issues with some packages")
        print("  Python 3.9 and below are not supported")
        return False
    return True


def install_requirements():
    """Install Python packages from requirements.txt."""
    print("\n" + "=" * 50)
    print("Installing Python packages...")
    print("=" * 50)

    requirements_file = Path(__file__).parent / "requirements.txt"
    if not requirements_file.exists():
        print("ERROR: requirements.txt not found")
        return False

    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("Packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to install packages: {e}")
        return False


def download_base_models():
    """Download OpenWakeWord base models (melspectrogram and embedding)."""
    print("\n" + "=" * 50)
    print("Downloading OpenWakeWord base models...")
    print("=" * 50)

    # Models directory
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)

    # Also create user directory for openwakeword
    user_models_dir = Path.home() / ".openwakeword" / "models"
    user_models_dir.mkdir(parents=True, exist_ok=True)

    # Model URLs from OpenWakeWord releases
    # These are the ONNX versions of the feature extraction models
    base_url = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1"

    models = {
        "melspectrogram.onnx": f"{base_url}/melspectrogram.onnx",
        "embedding_model.onnx": f"{base_url}/embedding_model.onnx",
    }

    for name, url in models.items():
        local_path = models_dir / name
        user_path = user_models_dir / name

        if local_path.exists() and local_path.stat().st_size > 100000:
            print(f"  {name}: already exists ({local_path.stat().st_size / 1024:.1f} KB)")
            # Also copy to user directory
            if not user_path.exists():
                import shutil
                shutil.copy2(local_path, user_path)
            continue

        print(f"  Downloading {name}...")
        try:
            urllib.request.urlretrieve(url, local_path)
            print(f"  {name}: downloaded ({local_path.stat().st_size / 1024:.1f} KB)")

            # Also copy to user's .openwakeword directory
            import shutil
            shutil.copy2(local_path, user_path)

        except Exception as e:
            print(f"  ERROR downloading {name}: {e}")
            print(f"  You may need to download manually from:")
            print(f"    {url}")
            return False

    print(f"\nModels saved to: {models_dir}")
    print(f"Also copied to: {user_models_dir}")
    return True


def check_ffmpeg():
    """Check if ffmpeg is installed (needed for MP3 conversion)."""
    print("\n" + "=" * 50)
    print("Checking ffmpeg installation...")
    print("=" * 50)

    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"  ffmpeg: {version_line}")
            return True
    except FileNotFoundError:
        pass

    print("  ffmpeg: NOT FOUND")
    print("\n  ffmpeg is required for MP3 to WAV conversion.")
    print("  Install it using:")
    print("    Windows: winget install ffmpeg")
    print("    macOS:   brew install ffmpeg")
    print("    Linux:   sudo apt install ffmpeg")
    return False


def verify_installation():
    """Verify all required packages can be imported."""
    print("\n" + "=" * 50)
    print("Verifying installation...")
    print("=" * 50)

    packages = [
        ("torch", "PyTorch"),
        ("torchaudio", "TorchAudio"),
        ("onnx", "ONNX"),
        ("onnxruntime", "ONNX Runtime"),
        ("soundfile", "SoundFile"),
        ("scipy", "SciPy"),
        ("numpy", "NumPy"),
        ("pydub", "PyDub"),
    ]

    all_ok = True
    for module, name in packages:
        try:
            __import__(module)
            print(f"  {name}: OK")
        except ImportError as e:
            print(f"  {name}: FAILED ({e})")
            all_ok = False

    # Check optional packages
    print("\nOptional packages (for TTS generation):")
    for module, name in [("openai", "OpenAI"), ("elevenlabs", "ElevenLabs")]:
        try:
            __import__(module)
            print(f"  {name}: OK")
        except ImportError:
            print(f"  {name}: not installed (optional)")

    return all_ok


def main():
    print("=" * 50)
    print("Wake Word Trainer Setup")
    print("=" * 50)

    # Change to script directory
    os.chdir(Path(__file__).parent)

    # Check Python version
    check_python_version()

    # Install packages
    if not install_requirements():
        print("\nSetup failed during package installation.")
        sys.exit(1)

    # Download models
    if not download_base_models():
        print("\nWarning: Could not download base models automatically.")
        print("You may need to download them manually.")

    # Check ffmpeg
    check_ffmpeg()

    # Verify installation
    if verify_installation():
        print("\n" + "=" * 50)
        print("Setup complete!")
        print("=" * 50)
        print("\nTo run the trainer:")
        print("  python app.py")
        print("\nOr on Windows, double-click run.bat")
    else:
        print("\nSetup completed with warnings. Some packages may need manual installation.")


if __name__ == "__main__":
    main()
