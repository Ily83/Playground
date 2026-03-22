# =============================================================================
# German Speaking Coach - Fully Private Local AI Agent
# =============================================================================
#
# A private, offline AI agent that helps you improve your German speaking
# skills.
#
# ARCHITECTURE OVERVIEW:
# ┌──────────────┐   ┌──────────────────┐   ┌──────────────────┐
# │  Microphone  │──▶│  faster-whisper   │──▶│                  │
# │ (sounddevice)│   │  (Local STT)     │   │   Ollama LLM     │
# └──────────────┘   └──────────────────┘   │   (gemma3 /      │
#                                           │    qwen3.5)      │
# ┌──────────────┐   ┌──────────────────┐   │                  │
# │   Speaker    │◀──│   Piper TTS      │◀──│  Corrects your   │
# │   (Output)   │   │  (Local Neural)  │   │  German & chats  │
# └──────────────┘   └──────────────────┘   └──────────────────┘
#
# SETUP INSTRUCTIONS:
# -------------------
# 1. Install Ollama (v0.18+):
#       https://ollama.com/download
#
# 2. Pull a German-capable model:
#       ollama pull gemma3
#       (alternatives: qwen3.5, llama3.1, deepseek-r1, mistral)
#
# 3. Install Python dependencies:
#       pip install ollama faster-whisper piper-tts sounddevice numpy scipy
#
# 4. Download a Piper German voice model:
#       mkdir -p piper_voices
#       cd piper_voices
#       wget https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/high/de_DE-thorsten-high.onnx
#       wget https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/high/de_DE-thorsten-high.onnx.json
#
# 5. Run this script:
#       python german_coach.py
# =============================================================================

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


@dataclass
class CoachConfig:
    """Central configuration — tweak everything from here."""

    # ---- LLM (Ollama v0.18+) ------------------------------------------------
    ollama_model: str = "gemma3"             # gemma3, qwen3.5, llama3.1, deepseek-r1
    ollama_base_url: str = "http://localhost:11434"

    # ---- Speech-to-Text (faster-whisper) ------------------------------------
    enable_voice_input: bool = True
    whisper_model_size: str = "base"          # tiny | base | small | medium | large-v3
    whisper_device: str = "cpu"               # cpu | cuda  (auto-detects if cuda available)
    whisper_compute_type: str = "int8"        # int8 (CPU) | float16 (GPU) | int8_float16

    # ---- Text-to-Speech (Piper TTS — local neural voices) -------------------
    enable_voice_output: bool = True
    piper_model_path: str = "./piper_voices/de_DE-thorsten-high.onnx"
    piper_config_path: str = "./piper_voices/de_DE-thorsten-high.onnx.json"

    # ---- Audio Recording -----------------------------------------------------
    sample_rate: int = 16000
    channels: int = 1
    silence_threshold: float = 0.02          # amplitude below this → silence
    silence_duration: float = 2.0            # seconds of silence to auto-stop
    max_record_seconds: int = 30

    # ---- Session / Learning --------------------------------------------------
    difficulty: str = "intermediate"         # beginner | intermediate | advanced
    focus_areas: List[str] = field(default_factory=lambda: [
        "grammar", "vocabulary", "pronunciation", "sentence_structure"
    ])
    log_dir: str = "./german_coach_logs"


# Data model

class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class ChatMessage:
    role: Role
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class SessionStats:
    total_user_messages: int = 0
    corrections_made: int = 0
    topics: List[str] = field(default_factory=list)
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None

# Prompt

SYSTEM_PROMPT_TEMPLATE = """\
You are **Deutsch-Coach**, a friendly and encouraging private German language tutor.

Your student's level: {difficulty}
Focus areas: {focus_areas}

## Your Rules
1. **Always reply in German first**, then provide an English explanation underneath
   (marked with 🇬🇧).
2. When the student writes/says something in German:
   - If it contains errors, quote the mistake, provide the correction, and briefly
     explain **why** it is wrong (grammar rule, gender, case, word order, etc.).
   - Format corrections like this:
     ❌  "<original>"
     ✅  "<corrected>"
     📖  <short explanation in English>
3. If the student's German is correct, praise them and continue the conversation.
4. Regularly introduce **new vocabulary** relevant to the conversation topic,
   formatted as:  🆕 <German word/phrase> — <English translation>
5. Occasionally ask the student questions to keep the conversation going.
6. Adapt your complexity to the student's level ({difficulty}).
7. If the student asks to practice a specific topic (e.g., Konjunktiv II, Dativ,
   Alltag, Beruf), steer the conversation in that direction.
8. Keep your answers concise but helpful — this is a conversation, not a lecture.
9. At the end of each response, provide a very short summary line:
   📊 Errors: <n> | New words: <n>
"""


def build_system_prompt(config: CoachConfig) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        difficulty=config.difficulty,
        focus_areas=", ".join(config.focus_areas),
    )

class SpeechToText:
    """Records audio from the microphone and transcribes with faster-whisper
    running 100% locally. Up to 4× faster than openai-whisper with less RAM."""

    def __init__(self, config: CoachConfig):
        self.config = config
        self._model = None

    def _load_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            print(f"⏳  Loading faster-whisper '{self.config.whisper_model_size}' model …")
            self._model = WhisperModel(
                self.config.whisper_model_size,
                device=self.config.whisper_device,
                compute_type=self.config.whisper_compute_type,
            )
            print("✅  Whisper model loaded.")

    # ---- recording -----------------------------------------------------------

    def record_audio(self) -> Optional[str]:
        """Record from mic until the user presses Enter; return path to .wav file."""
        import sounddevice as sd
        import numpy as np
        import threading

        cfg = self.config
        print("\n🎤  Recording … (press Enter when you're done speaking)")

        frames: List[np.ndarray] = []
        stop_flag = [False]
        chunk_duration = 0.1                          # 100 ms chunks
        chunk_samples = int(cfg.sample_rate * chunk_duration)

        def _wait_for_enter():
            input()                                   # blocks until Enter
            stop_flag[0] = True

        threading.Thread(target=_wait_for_enter, daemon=True).start()

        try:
            while not stop_flag[0]:
                chunk = sd.rec(
                    chunk_samples,
                    samplerate=cfg.sample_rate,
                    channels=cfg.channels,
                    dtype="float32",
                )
                sd.wait()
                frames.append(chunk)

                total_seconds = len(frames) * chunk_duration
                if total_seconds >= cfg.max_record_seconds:
                    print("⏱  Max recording time reached.")
                    break

        except KeyboardInterrupt:
            print("\n⏹  Recording stopped.")

        if not frames:
            return None

        audio_data = np.concatenate(frames, axis=0)

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(cfg.channels)
            wf.setsampwidth(2)                        # 16-bit
            wf.setframerate(cfg.sample_rate)
            wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())

        duration = len(audio_data) / cfg.sample_rate
        print(f"📝  Recorded {duration:.1f}s of audio.")
        return tmp.name

    # ---- transcription -------------------------------------------------------

    def transcribe(self, audio_path: str) -> str:
        """Transcribe a .wav file to text using faster-whisper."""
        self._load_model()
        segments, info = self._model.transcribe(
            audio_path,
            language="de",
            beam_size=5,
            vad_filter=False, # filter silcence
            condition_on_previous_text=True,
            initial_prompt="Ich lerne Deutsch. "  
        )
        text = " ".join(segment.text.strip() for segment in segments)
        try:
            os.unlink(audio_path)
        except OSError:
            pass
        return text

    def listen(self) -> Optional[str]:
        """Record and transcribe in one step."""
        audio_path = self.record_audio()
        if audio_path is None:
            return None
        text = self.transcribe(audio_path)
        if text:
            print(f"🗣  You said: {text}")
        return text


# Piper TTS | Text to speech
class TextToSpeech:
    """Speaks German text aloud using Piper TTS"""

    def __init__(self, config: CoachConfig):
        self.config = config
        self._voice = None

    def _init_voice(self):
        if self._voice is None:
            from piper import PiperVoice
            model_path = self.config.piper_model_path
            config_path = self.config.piper_config_path

            if not Path(model_path).exists():
                print(f"\nPiper voice model not found at: {model_path}")
                print("   Download a German voice:")
                print("   mkdir -p piper_voices && cd piper_voices")
                print("   wget https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/high/de_DE-thorsten-high.onnx")
                print("   wget https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/high/de_DE-thorsten-high.onnx.json")
                print("\n   Or browse voices: https://github.com/rhasspy/piper/blob/master/VOICES.md")
                print("   ⏭  Continuing without voice output.\n")
                self.config.enable_voice_output = False
                return

            print("⏳  Loading Piper TTS voice …")
            self._voice = PiperVoice.load(model_path, config_path=config_path)
            print("✅  Piper TTS ready.")

    def speak(self, text: str):
        """Speak the given text aloud using Piper neural TTS."""
        if not self.config.enable_voice_output:
            return

        try:
            self._init_voice()
            if self._voice is None:
                return
            
            german_part = text.split("📊")[0] if "📊" in text else text
            print(f"debug: printing the text {german_part}")
            clean = german_part
            for char in ["**", "*", "❌", "✅", "📖", "🆕", "📊", "#"]:
                clean = clean.replace(char, "")
            clean = clean.strip()
            
            if not clean:
                return

            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            with wave.open(tmp.name, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self._voice.config.sample_rate)

                
                if hasattr(self._voice, "synthesize_wav"):
                    self._voice.synthesize_wav(clean, wav_file)
                else:
                    for chunk in self._voice.synthesize(clean):
                        wav_file.writeframes(chunk.audio_int16_bytes)

            import sounddevice as sd
            from scipy.io import wavfile

            sr, audio = wavfile.read(tmp.name)
            sd.play(audio, samplerate=sr)
            sd.wait()

            os.unlink(tmp.name)

        except Exception as e:
            print(f"🔇  TTS error (non-critical): {e}")


# local llm client (Ollama v0.18+)

class LocalLLM:
    """Communicates with the Ollama API running locally."""

    def __init__(self, config: CoachConfig):
        self.config = config

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Send messages to Ollama and return the assistant response."""
        import ollama

        response = ollama.chat(
            model=self.config.ollama_model,
            messages=messages,
            options={
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 1024,
            },
        )
        return response["message"]["content"]

    def check_connection(self) -> bool:
        """Verify that Ollama is running and the model is available."""
        try:
            import ollama
            models = ollama.list()
            available = [m["model"] for m in models.get("models", [])]
            model_base = self.config.ollama_model.split(":")[0]
            found = any(model_base in m for m in available)
            if not found:
                print(f"⚠️  Model '{self.config.ollama_model}' not found.")
                print(f"   Available models: {available}")
                print(f"   Run: ollama pull {self.config.ollama_model}")
            return found
        except Exception as e:
            print(f"❌  Cannot connect to Ollama: {e}")
            print("   Make sure Ollama is running: ollama serve")
            return False


# Session logger — saves your progress locally
class SessionLogger:
    """Persists conversation logs and stats to local JSON files."""

    def __init__(self, config: CoachConfig):
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.session_file = self.log_dir / f"session_{datetime.now():%Y%m%d_%H%M%S}.json"
        self.stats = SessionStats()

    def log_message(self, message: ChatMessage):
        if message.role == Role.USER:
            self.stats.total_user_messages += 1

    def log_correction(self):
        self.stats.corrections_made += 1

    def save(self, history: List[ChatMessage]):
        """Save the full session to disk."""
        self.stats.end_time = datetime.now().isoformat()
        data = {
            "stats": {
                "total_user_messages": self.stats.total_user_messages,
                "corrections_made": self.stats.corrections_made,
                "start_time": self.stats.start_time,
                "end_time": self.stats.end_time,
            },
            "messages": [
                {"role": m.role.value, "content": m.content, "timestamp": m.timestamp}
                for m in history
            ],
        }
        self.session_file.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(f"\n💾  Session saved to {self.session_file}")


class GermanCoach:
    """Orchestrates the full conversation loop."""

    def __init__(self, config: Optional[CoachConfig] = None):
        self.config = config or CoachConfig()
        self.llm = LocalLLM(self.config)
        self.stt = SpeechToText(self.config) if self.config.enable_voice_input else None
        self.tts = TextToSpeech(self.config) if self.config.enable_voice_output else None
        self.logger = SessionLogger(self.config)
        self.history: List[ChatMessage] = []

        # Initialize with system prompt
        system_msg = ChatMessage(
            role=Role.SYSTEM,
            content=build_system_prompt(self.config),
        )
        self.history.append(system_msg)

    # ---- helpers -------------------------------------------------------------

    def _history_for_llm(self) -> List[Dict[str, str]]:
        """Convert internal history to the dict format Ollama expects."""
        return [{"role": m.role.value, "content": m.content} for m in self.history]

    def _count_corrections(self, response: str) -> int:
        """Rough heuristic: count ❌ markers to detect corrections."""
        return response.count("❌")

    # ---- input methods -------------------------------------------------------

    def get_user_input(self) -> Optional[str]:
        """Get input via voice OR text, depending on user choice."""
        if self.stt and self.config.enable_voice_input:
            print("\n[V] Speak  |  [T] Type  |  [Q] Quit")
            choice = input("→ ").strip().lower()
            if choice == "q":
                return None
            elif choice == "v":
                return self.stt.listen()
            else:
                return input("✏️  Type in German: ").strip()
        else:
            print()
            text = input("✏️  You (or 'quit'): ").strip()
            return None if text.lower() in ("quit", "exit", "q") else text

    # ---- main loop -----------------------------------------------------------

    def run(self):
        """Start the interactive coaching session."""
        self._print_banner()

        # Verify Ollama is running
        if not self.llm.check_connection():
            sys.exit(1)

        # Kick off with a greeting from the coach
        greeting = self._get_response(
            "Please greet the student in German and suggest a conversation topic "
            "appropriate for their level. Keep it warm and short."
        )
        print(f"\n🤖 Coach: {greeting}")
        if self.tts:
            self.tts.speak(greeting)

        # Conversation loop
        while True:
            user_text = self.get_user_input()
            if user_text is None:                     # explicit Q / quit
                print("\n👋  Tschüss! Bis zum nächsten Mal!")
                break
            if not user_text.strip():                 # empty transcription — retry
                print("🔁  Didn't catch that, please try again.")
                continue

            # Add user message
            user_msg = ChatMessage(role=Role.USER, content=user_text)
            self.history.append(user_msg)
            self.logger.log_message(user_msg)

            # Get coach response
            response = self._get_response(user_text)
            print(f"\n🤖 Coach: {response}")

            # Track corrections
            n_corrections = self._count_corrections(response)
            for _ in range(n_corrections):
                self.logger.log_correction()

            # Speak the response
            if self.tts:
                self.tts.speak(response)

        # Save session
        self.logger.save(self.history)
        self._print_summary()

    def _get_response(self, user_input: str) -> str:
        """Query the local LLM and record the response."""
        messages = self._history_for_llm()
        response_text = self.llm.chat(messages)

        assistant_msg = ChatMessage(role=Role.ASSISTANT, content=response_text)
        self.history.append(assistant_msg)
        return response_text


    def _print_banner(self):
        banner = """
        German Speaking Coach
        Level: {level:<15s}                                        ║
        Model: {model:<15s}                                        
        """.format(level=self.config.difficulty.upper(), model=self.config.ollama_model)
        print(banner)

    def _print_summary(self):
        stats = self.logger.stats
        print("\n" + "=" * 50)
        print("📊  SESSION SUMMARY")
        print("=" * 50)
        print(f"  Messages sent     : {stats.total_user_messages}")
        print(f"  Corrections made  : {stats.corrections_made}")
        print(f"  Session duration  : {stats.start_time} → {stats.end_time}")
        print("=" * 50)


if __name__ == "__main__":
    # ---------- Customize your config here ----------
    config = CoachConfig(
        ollama_model="gemma3",            # Recommended: gemma3, qwen3.5, llama3.1
        difficulty="intermediate",        # beginner / intermediate / advanced
        enable_voice_input=True,          # Set False for text-only mode
        enable_voice_output=True,         # Set False to disable spoken responses
        whisper_model_size="small",        # Larger = more accurate but slower
        whisper_device="cpu",             # Use "cuda" if you have an NVIDIA GPU
        whisper_compute_type="int8",      # Use "float16" for GPU
        piper_model_path="./piper_voices/de_DE-thorsten-high.onnx",
        piper_config_path="./piper_voices/de_DE-thorsten-high.onnx.json",
    )

    coach = GermanCoach(config)
    coach.run()