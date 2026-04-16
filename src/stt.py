from pathlib import Path
from groq import Groq
from src.config import (
    GROQ_API_KEY,
    STT_MODEL,
    SUPPORTED_AUDIO_FORMATS,
    MAX_AUDIO_SIZE_MB,
)

client = Groq(api_key=GROQ_API_KEY)


def validate_audio(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() not in SUPPORTED_AUDIO_FORMATS:
        raise ValueError(f"Unsupported format: {path.suffix}")

    if path.stat().st_size > MAX_AUDIO_SIZE_MB * 1024 * 1024:
        raise ValueError(f"File too large (> {MAX_AUDIO_SIZE_MB}MB)")


def transcribe(audio_path: str | Path, language: str = "en") -> dict:
    path = Path(audio_path)
    result = {
        "success": False,
        "text": "",
        "language": language,
        "duration": None,
        "error": None,
    }

    try:
        validate_audio(path)

        with open(path, "rb") as f:
            resp = client.audio.transcriptions.create(
                model=STT_MODEL,
                file=f,
                response_format="verbose_json",
                language=language,
            )

        result.update(
            {
                "success": True,
                "text": resp.text.strip(),
                "language": getattr(resp, "language", language),
                "duration": getattr(resp, "duration", None),
            }
        )

    except Exception as e:
        result["error"] = str(e)

    return result


def transcribe_text(audio_path: str | Path, language: str = "en") -> str:
    res = transcribe(audio_path, language)
    if not res["success"]:
        raise RuntimeError(res["error"])
    return res["text"]
