

import os
import sys
import json
import tempfile
import wave
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from enum import Enum
from piper import PiperVoice
from scipy.io import wavfile
import sounddevice as sd
import soundfile as sf
        
piper_model_path: str = "./piper_voices/de_DE-thorsten-high.onnx"
piper_config_path: str = "./piper_voices/de_DE-thorsten-high.onnx.json"

voice = PiperVoice.load(piper_model_path, config_path=piper_config_path)
import sounddevice as sd
import numpy as np

def test_audio():
    sr = 22050
    t = np.linspace(0,1,sr)
    tone = (np.sin(2* np.pi * t) * 32767).astype('int16')
    sd.play(tone, samplerate=sr)
    sd.wait()
    print("fertig")
    


def speak(text: str):
    try:
        german_part = text.split("🇬🇧")[0] if "🇬🇧" in text else text
        print(f"debug: printing the text {german_part}")
        # Remove markdown and emojis for cleaner speech
        clean = german_part
        for char in ["**", "*", "❌", "✅", "📖", "🆕", "📊", "#"]:
            clean = clean.replace(char, "")
        clean = clean.strip()

        if not clean:
            return

        # Wave params must be set BEFORE synthesize() writes frames
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()

        with wave.open(tmp_path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(voice.config.sample_rate)

            
            if hasattr(voice, "synthesize_wav"):
                voice.synthesize_wav(clean, wav_file)
            else:
                # synthesize() is a generator yielding AudioChunk objects
                for chunk in voice.synthesize(clean):
                    wav_file.writeframes(chunk.audio_int16_bytes)
    


        data, sr = sf.read(tmp_path, dtype='float32')
        silence = np.zeros(int(sr * 0.11), dtype=np.float32)
        data = np.concatenate([silence, data])
        print("Frames:", len(data), "SR:", sr)
        sd.play(data, samplerate=sr)
        sd.wait()
        os.unlink(tmp_path)
    
    except Exception as e:
      print(f"🔇  TTS error (non-critical): {e}")

speak("Hallo ich heisse Patrik.")
# print(sd.query_devices())
# print("Default output:", sd.default.device)
# test_audio()