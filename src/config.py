import os
from pathlib import Path
from dotenv import load_dotenv

# Load env once
load_dotenv()

# Project Paths
BASE_DIR = Path(__file__).parent.parent
SRC_DIR = BASE_DIR / "src"
OUTPUT_DIR = BASE_DIR / "output"
TESTS_DIR = BASE_DIR / "tests"

# Ensure output dir exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Models
STT_MODEL = "whisper-large-v3"
LLM_MODEL = "llama-3.3-70b-versatile"
INTENT_MODEL = "llama-3.3-70b-versatile"

# Settings
MAX_AUDIO_SIZE_MB = 25
SUPPORTED_AUDIO_FORMATS = [".wav", ".mp3", ".mp4", ".m4a", ".ogg", ".flac", ".webm"]

# Server Settings
SERVER_NAME = os.getenv("SERVER_NAME", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "7860"))
