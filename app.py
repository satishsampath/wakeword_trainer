#!/usr/bin/env python3
"""
Wake Word Training Tool

A simple GUI application for generating TTS training data and training
custom wake word detection models.
"""

import os
import sys
import json
import threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from pathlib import Path
import subprocess

# Configuration file path
CONFIG_FILE = Path(__file__).parent / "user_config.json"
DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "output"


class WakeWordTrainerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Wake Word Trainer")
        self.root.geometry("700x850")
        self.root.resizable(True, True)

        # Load saved config
        self.config = self.load_config()

        # Create main container with padding
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")

        # Configure grid weights for resizing
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        row = 0

        # === Wake Word Section ===
        ttk.Label(main_frame, text="Wake Word Configuration", font=("", 12, "bold")).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(0, 10)
        )
        row += 1

        ttk.Label(main_frame, text="Wake Word:").grid(row=row, column=0, sticky="w", pady=5)
        self.wake_word_var = tk.StringVar(value=self.config.get("wake_word", "Hey Assistant"))
        self.wake_word_entry = ttk.Entry(main_frame, textvariable=self.wake_word_var, width=40)
        self.wake_word_entry.grid(row=row, column=1, sticky="ew", pady=5, padx=(5, 0))
        row += 1

        ttk.Label(main_frame, text="Model Name:").grid(row=row, column=0, sticky="w", pady=5)
        self.model_name_var = tk.StringVar(value=self.config.get("model_name", "hey_assistant"))
        self.model_name_entry = ttk.Entry(main_frame, textvariable=self.model_name_var, width=40)
        self.model_name_entry.grid(row=row, column=1, sticky="ew", pady=5, padx=(5, 0))
        ttk.Label(main_frame, text="(lowercase, underscores)", foreground="gray").grid(
            row=row, column=2, sticky="w", padx=5
        )
        row += 1

        # Separator
        ttk.Separator(main_frame, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=15
        )
        row += 1

        # === API Keys Section ===
        ttk.Label(main_frame, text="API Keys", font=("", 12, "bold")).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(0, 10)
        )
        row += 1

        ttk.Label(main_frame, text="OpenAI API Key:").grid(row=row, column=0, sticky="w", pady=5)
        self.openai_key_var = tk.StringVar(value=self.config.get("openai_api_key", ""))
        self.openai_key_entry = ttk.Entry(main_frame, textvariable=self.openai_key_var, width=40, show="*")
        self.openai_key_entry.grid(row=row, column=1, sticky="ew", pady=5, padx=(5, 0))
        self.show_openai_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(main_frame, text="Show", variable=self.show_openai_var,
                       command=lambda: self.toggle_show(self.openai_key_entry, self.show_openai_var)).grid(
            row=row, column=2, padx=5
        )
        row += 1

        ttk.Label(main_frame, text="ElevenLabs API Key:").grid(row=row, column=0, sticky="w", pady=5)
        self.elevenlabs_key_var = tk.StringVar(value=self.config.get("elevenlabs_api_key", ""))
        self.elevenlabs_key_entry = ttk.Entry(main_frame, textvariable=self.elevenlabs_key_var, width=40, show="*")
        self.elevenlabs_key_entry.grid(row=row, column=1, sticky="ew", pady=5, padx=(5, 0))
        self.show_elevenlabs_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(main_frame, text="Show", variable=self.show_elevenlabs_var,
                       command=lambda: self.toggle_show(self.elevenlabs_key_entry, self.show_elevenlabs_var)).grid(
            row=row, column=2, padx=5
        )
        row += 1

        # Separator
        ttk.Separator(main_frame, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=15
        )
        row += 1

        # === TTS Generation Section ===
        ttk.Label(main_frame, text="Generate Training Audio", font=("", 12, "bold")).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(0, 10)
        )
        row += 1

        # OpenAI options
        self.use_openai_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(main_frame, text="Use OpenAI TTS", variable=self.use_openai_var).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=2
        )
        row += 1

        ttk.Label(main_frame, text="  Voices:").grid(row=row, column=0, sticky="w")
        self.openai_voices_var = tk.StringVar(value="alloy, echo, fable, nova, onyx, shimmer")
        ttk.Entry(main_frame, textvariable=self.openai_voices_var, width=40).grid(
            row=row, column=1, sticky="ew", padx=(5, 0)
        )
        row += 1

        ttk.Label(main_frame, text="  Speed variations:").grid(row=row, column=0, sticky="w")
        self.openai_speeds_var = tk.StringVar(value="0.9, 0.95, 1.0, 1.05, 1.1")
        ttk.Entry(main_frame, textvariable=self.openai_speeds_var, width=40).grid(
            row=row, column=1, sticky="ew", padx=(5, 0)
        )
        row += 1

        # ElevenLabs options
        self.use_elevenlabs_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(main_frame, text="Use ElevenLabs TTS", variable=self.use_elevenlabs_var).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(10, 2)
        )
        row += 1

        ttk.Label(main_frame, text="  Voice count:").grid(row=row, column=0, sticky="w")
        self.elevenlabs_count_var = tk.StringVar(value="10")
        ttk.Entry(main_frame, textvariable=self.elevenlabs_count_var, width=10).grid(
            row=row, column=1, sticky="w", padx=(5, 0)
        )
        ttk.Label(main_frame, text="(uses random voices from library)", foreground="gray").grid(
            row=row, column=2, sticky="w", padx=5
        )
        row += 1

        # Generate button
        self.generate_btn = ttk.Button(main_frame, text="Generate TTS Audio", command=self.generate_audio)
        self.generate_btn.grid(row=row, column=0, columnspan=3, pady=15)
        row += 1

        # Separator
        ttk.Separator(main_frame, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=15
        )
        row += 1

        # === Training Section ===
        ttk.Label(main_frame, text="Train Model", font=("", 12, "bold")).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(0, 10)
        )
        row += 1

        # Training type selection
        ttk.Label(main_frame, text="Training type:").grid(row=row, column=0, sticky="w", pady=5)
        self.training_type_var = tk.StringVar(value="openwakeword")
        training_frame = ttk.Frame(main_frame)
        training_frame.grid(row=row, column=1, columnspan=2, sticky="w", padx=(5, 0))
        ttk.Radiobutton(training_frame, text="OpenWakeWord (recommended)",
                       variable=self.training_type_var, value="openwakeword").pack(side="left")
        ttk.Radiobutton(training_frame, text="Simple CNN",
                       variable=self.training_type_var, value="simple").pack(side="left", padx=(20, 0))
        row += 1

        # Augmentation option
        self.augment_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(main_frame, text="Apply data augmentation (noise, reverb, speed, volume)",
                       variable=self.augment_var).grid(row=row, column=0, columnspan=3, sticky="w", pady=5)
        row += 1

        # Train button
        self.train_btn = ttk.Button(main_frame, text="Train Model", command=self.train_model)
        self.train_btn.grid(row=row, column=0, columnspan=3, pady=15)
        row += 1

        # Separator
        ttk.Separator(main_frame, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=15
        )
        row += 1

        # === Export Section ===
        ttk.Label(main_frame, text="Export Model", font=("", 12, "bold")).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(0, 10)
        )
        row += 1

        ttk.Label(main_frame, text="Export the trained model to ONNX format for deployment.", foreground="gray").grid(
            row=row, column=0, columnspan=3, sticky="w"
        )
        row += 1

        # Export button
        self.export_btn = ttk.Button(main_frame, text="Export to ONNX", command=self.export_model)
        self.export_btn.grid(row=row, column=0, columnspan=3, pady=15)
        row += 1

        # Separator
        ttk.Separator(main_frame, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=15
        )
        row += 1

        # === Log Section ===
        ttk.Label(main_frame, text="Log", font=("", 12, "bold")).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(0, 5)
        )
        row += 1

        self.log_text = scrolledtext.ScrolledText(main_frame, height=12, width=80, state="disabled")
        self.log_text.grid(row=row, column=0, columnspan=3, sticky="nsew", pady=5)
        main_frame.rowconfigure(row, weight=1)
        row += 1

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief="sunken", anchor="w")
        status_bar.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(10, 0))

        # Save config on close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Log startup
        self.log("Wake Word Trainer started")
        self.log(f"Data directory: {DATA_DIR}")
        self.log(f"Output directory: {OUTPUT_DIR}")

    def toggle_show(self, entry, var):
        entry.config(show="" if var.get() else "*")

    def load_config(self):
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r") as f:
                    return json.load(f)
            except:
                pass
        return {}

    def save_config(self):
        config = {
            "wake_word": self.wake_word_var.get(),
            "model_name": self.model_name_var.get(),
            "openai_api_key": self.openai_key_var.get(),
            "elevenlabs_api_key": self.elevenlabs_key_var.get(),
        }
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)

    def on_close(self):
        self.save_config()
        self.root.destroy()

    def log(self, message):
        self.log_text.config(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")
        self.root.update_idletasks()

    def set_status(self, message):
        self.status_var.set(message)
        self.root.update_idletasks()

    def update_config_py(self):
        """Update config.py with current settings."""
        wake_word = self.wake_word_var.get()
        model_name = self.model_name_var.get()

        config_path = Path(__file__).parent / "config.py"

        # Read current config
        with open(config_path, "r") as f:
            content = f.read()

        # Update WAKE_WORD and MODEL_NAME
        import re
        content = re.sub(r'WAKE_WORD = ".*?"', f'WAKE_WORD = "{wake_word}"', content)
        content = re.sub(r'MODEL_NAME = ".*?"', f'MODEL_NAME = "{model_name}"', content)

        with open(config_path, "w") as f:
            f.write(content)

    def generate_audio(self):
        """Generate TTS audio files."""
        wake_word = self.wake_word_var.get().strip()
        if not wake_word:
            messagebox.showerror("Error", "Please enter a wake word")
            return

        use_openai = self.use_openai_var.get()
        use_elevenlabs = self.use_elevenlabs_var.get()

        if use_openai and not self.openai_key_var.get().strip():
            messagebox.showerror("Error", "Please enter OpenAI API key")
            return

        if use_elevenlabs and not self.elevenlabs_key_var.get().strip():
            messagebox.showerror("Error", "Please enter ElevenLabs API key")
            return

        if not use_openai and not use_elevenlabs:
            messagebox.showerror("Error", "Please select at least one TTS provider")
            return

        # Disable button during generation
        self.generate_btn.config(state="disabled")
        self.set_status("Generating audio...")

        # Run in background thread
        thread = threading.Thread(target=self._generate_audio_thread)
        thread.daemon = True
        thread.start()

    def _generate_audio_thread(self):
        try:
            wake_word = self.wake_word_var.get().strip()
            positive_dir = DATA_DIR / "positive"
            positive_dir.mkdir(parents=True, exist_ok=True)

            total_files = 0

            # Generate OpenAI TTS
            if self.use_openai_var.get():
                self.log("\n=== Generating OpenAI TTS ===")
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=self.openai_key_var.get().strip())

                    voices = [v.strip() for v in self.openai_voices_var.get().split(",")]
                    speeds = [float(s.strip()) for s in self.openai_speeds_var.get().split(",")]

                    for voice in voices:
                        for speed in speeds:
                            try:
                                self.log(f"Generating: {voice} @ {speed}x...")
                                response = client.audio.speech.create(
                                    model="tts-1-hd",
                                    voice=voice,
                                    input=wake_word,
                                    speed=speed
                                )

                                filename = f"openai_{voice}_{speed:.2f}.mp3"
                                filepath = positive_dir / filename
                                response.stream_to_file(str(filepath))
                                total_files += 1

                            except Exception as e:
                                self.log(f"  Error: {e}")

                    self.log(f"OpenAI: Generated {total_files} files")

                except ImportError:
                    self.log("Error: openai package not installed. Run: pip install openai")
                except Exception as e:
                    self.log(f"OpenAI error: {e}")

            # Generate ElevenLabs TTS
            if self.use_elevenlabs_var.get():
                self.log("\n=== Generating ElevenLabs TTS ===")
                try:
                    from elevenlabs.client import ElevenLabs

                    client = ElevenLabs(api_key=self.elevenlabs_key_var.get().strip())

                    # Get available voices
                    voices_response = client.voices.get_all()
                    voices = voices_response.voices

                    count = int(self.elevenlabs_count_var.get())
                    count = min(count, len(voices))

                    self.log(f"Found {len(voices)} voices, using {count}")

                    elevenlabs_count = 0
                    for i, voice in enumerate(voices[:count]):
                        try:
                            self.log(f"Generating: {voice.name}...")

                            audio = client.generate(
                                text=wake_word,
                                voice=voice.voice_id,
                                model="eleven_multilingual_v2"
                            )

                            filename = f"elevenlabs_{voice.voice_id[:8]}.mp3"
                            filepath = positive_dir / filename

                            with open(filepath, "wb") as f:
                                for chunk in audio:
                                    f.write(chunk)

                            elevenlabs_count += 1
                            total_files += 1

                        except Exception as e:
                            self.log(f"  Error: {e}")

                    self.log(f"ElevenLabs: Generated {elevenlabs_count} files")

                except ImportError:
                    self.log("Error: elevenlabs package not installed. Run: pip install elevenlabs")
                except Exception as e:
                    self.log(f"ElevenLabs error: {e}")

            # Convert MP3 to WAV
            self.log("\n=== Converting to WAV ===")
            self._convert_mp3_to_wav(positive_dir)

            self.log(f"\nDone! Generated {total_files} audio files")
            self.log(f"Files saved to: {positive_dir}")

            self.root.after(0, lambda: self.set_status(f"Generated {total_files} files"))
            self.root.after(0, lambda: messagebox.showinfo("Success", f"Generated {total_files} audio files"))

        except Exception as e:
            self.log(f"Error: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.root.after(0, lambda: self.generate_btn.config(state="normal"))

    def _convert_mp3_to_wav(self, directory):
        """Convert MP3 files to WAV format."""
        try:
            import torchaudio

            mp3_files = list(directory.glob("*.mp3"))
            for mp3_path in mp3_files:
                try:
                    wav_path = mp3_path.with_suffix(".wav")

                    # Load and convert
                    waveform, sr = torchaudio.load(str(mp3_path))

                    # Resample to 16kHz if needed
                    if sr != 16000:
                        resampler = torchaudio.transforms.Resample(sr, 16000)
                        waveform = resampler(waveform)

                    # Convert to mono
                    if waveform.shape[0] > 1:
                        waveform = waveform.mean(dim=0, keepdim=True)

                    # Save as WAV
                    torchaudio.save(str(wav_path), waveform, 16000)
                    self.log(f"Converted: {mp3_path.name} -> {wav_path.name}")

                except Exception as e:
                    self.log(f"  Error converting {mp3_path.name}: {e}")

        except ImportError:
            self.log("Warning: torchaudio not installed, skipping MP3 to WAV conversion")
            self.log("Run: pip install torchaudio")

    def train_model(self):
        """Train the wake word model."""
        wake_word = self.wake_word_var.get().strip()
        model_name = self.model_name_var.get().strip()

        if not wake_word:
            messagebox.showerror("Error", "Please enter a wake word")
            return

        if not model_name:
            messagebox.showerror("Error", "Please enter a model name")
            return

        # Check for training data
        positive_dir = DATA_DIR / "positive"
        wav_files = list(positive_dir.glob("*.wav"))

        if len(wav_files) == 0:
            messagebox.showerror("Error", f"No WAV files found in {positive_dir}\nPlease generate audio first.")
            return

        # Update config.py
        self.update_config_py()

        # Disable button during training
        self.train_btn.config(state="disabled")
        self.set_status("Training model...")

        # Run in background thread
        thread = threading.Thread(target=self._train_model_thread)
        thread.daemon = True
        thread.start()

    def _train_model_thread(self):
        try:
            training_type = self.training_type_var.get()

            if training_type == "openwakeword":
                script = "train_openwakeword.py"
            else:
                script = "train_simple.py"

            script_path = Path(__file__).parent / script

            self.log(f"\n=== Training with {script} ===")
            self.log(f"Wake word: {self.wake_word_var.get()}")
            self.log(f"Model name: {self.model_name_var.get()}")

            # Apply augmentation if selected
            if self.augment_var.get():
                self.log("\nApplying data augmentation...")
                self._augment_data()

            # Run training script
            self.log("\nStarting training...\n")

            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(Path(__file__).parent)
            )

            for line in iter(process.stdout.readline, ''):
                self.log(line.rstrip())

            process.wait()

            if process.returncode == 0:
                self.log("\nTraining complete!")
                output_dir = OUTPUT_DIR
                self.log(f"Model saved to: {output_dir}")
                self.root.after(0, lambda: self.set_status("Training complete"))
                self.root.after(0, lambda: messagebox.showinfo("Success", f"Model trained successfully!\nSaved to: {output_dir}"))
            else:
                self.log(f"\nTraining failed with code {process.returncode}")
                self.root.after(0, lambda: self.set_status("Training failed"))
                self.root.after(0, lambda: messagebox.showerror("Error", "Training failed. Check the log for details."))

        except Exception as e:
            self.log(f"Error: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.root.after(0, lambda: self.train_btn.config(state="normal"))

    def _augment_data(self):
        """Apply data augmentation to training files."""
        try:
            import torchaudio
            import torch
            import random

            positive_dir = DATA_DIR / "positive"
            wav_files = list(positive_dir.glob("*.wav"))

            augmented_count = 0

            for wav_path in wav_files:
                # Skip already augmented files
                if "_aug_" in wav_path.name:
                    continue

                try:
                    waveform, sr = torchaudio.load(str(wav_path))
                    base_name = wav_path.stem

                    # Noise augmentation
                    noise_level = random.uniform(0.002, 0.01)
                    noisy = waveform + torch.randn_like(waveform) * noise_level
                    noisy_path = positive_dir / f"{base_name}_aug_noise.wav"
                    torchaudio.save(str(noisy_path), noisy, sr)
                    augmented_count += 1

                    # Volume augmentation
                    gain = random.uniform(0.7, 1.3)
                    vol_adjusted = waveform * gain
                    vol_path = positive_dir / f"{base_name}_aug_vol.wav"
                    torchaudio.save(str(vol_path), vol_adjusted, sr)
                    augmented_count += 1

                except Exception as e:
                    self.log(f"  Augmentation error for {wav_path.name}: {e}")

            self.log(f"Created {augmented_count} augmented samples")

        except ImportError:
            self.log("Warning: torchaudio not installed, skipping augmentation")

    def export_model(self):
        """Export the trained model to ONNX format."""
        model_name = self.model_name_var.get().strip()

        if not model_name:
            messagebox.showerror("Error", "Please enter a model name")
            return

        # Check for trained model
        pt_file = OUTPUT_DIR / f"{model_name}.pt"
        if not pt_file.exists():
            messagebox.showerror("Error", f"No trained model found: {pt_file}\nPlease train the model first.")
            return

        # Update config.py
        self.update_config_py()

        # Disable button during export
        self.export_btn.config(state="disabled")
        self.set_status("Exporting model...")

        # Run in background thread
        thread = threading.Thread(target=self._export_model_thread)
        thread.daemon = True
        thread.start()

    def _export_model_thread(self):
        try:
            model_name = self.model_name_var.get().strip()

            self.log(f"\n=== Exporting Model to ONNX ===")
            self.log(f"Model name: {model_name}")

            # Run export script
            script_path = Path(__file__).parent / "export_onnx.py"

            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(Path(__file__).parent)
            )

            for line in iter(process.stdout.readline, ''):
                self.log(line.rstrip())

            process.wait()

            if process.returncode == 0:
                onnx_path = OUTPUT_DIR / f"{model_name}.onnx"
                self.log(f"\nExport complete!")
                self.log(f"ONNX model: {onnx_path}")

                if onnx_path.exists():
                    size_kb = onnx_path.stat().st_size / 1024
                    self.log(f"Size: {size_kb:.1f} KB")

                self.root.after(0, lambda: self.set_status("Export complete"))
                self.root.after(0, lambda: messagebox.showinfo("Success", f"Model exported successfully!\n\n{onnx_path}"))
            else:
                self.log(f"\nExport failed with code {process.returncode}")
                self.root.after(0, lambda: self.set_status("Export failed"))
                self.root.after(0, lambda: messagebox.showerror("Error", "Export failed. Check the log for details."))

        except Exception as e:
            self.log(f"Error: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.root.after(0, lambda: self.export_btn.config(state="normal"))


def main():
    # Ensure directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "positive").mkdir(exist_ok=True)
    (DATA_DIR / "negative").mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    root = tk.Tk()
    app = WakeWordTrainerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
