from src.stt import transcribe, transcribe_text
from pathlib import Path
from unittest.mock import patch, MagicMock

AUDIO_DIR = Path(__file__).parent / "audio"
TEST_FILE = AUDIO_DIR / "test_code.mp3"


def run_tests():
    print("=== STT Tests ===")

    # Ensure audio dir and a dummy test file exist for the test logic to pass
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    if not TEST_FILE.exists():
        TEST_FILE.write_text("dummy audio data")

    with patch("src.stt.client.audio.transcriptions.create") as mock_transcribe:
        # Prepare mock response
        mock_resp = MagicMock()
        mock_resp.text = "This is a mocked transcription."
        mock_transcribe.return_value = mock_resp

        # 1. Valid audio
        res = transcribe(TEST_FILE)
        print(f"Debug: {res}")
        assert res["success"], "Transcription failed"
        assert len(res["text"]) > 0, "Text should not be empty"

        # 2. Missing file
        res = transcribe("nonexistent.wav")
        assert not res["success"], "Missing file should fail"

        # 3. transcribe_text helper
        result_text = transcribe_text(TEST_FILE)
        assert isinstance(result_text, str)
        assert len(result_text) > 0

    print("\nAll STT tests passed!")


if __name__ == "__main__":
    run_tests()
