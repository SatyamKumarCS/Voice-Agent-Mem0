import json
from unittest.mock import patch, MagicMock
from src.intent import classify_compound_intent

TEST_CASES = [
    ("Create a new file called notes.txt", ["create_file"]),
    ("Write Python code for a retry function", ["write_code"]),
    ("Summarize this article about AI", ["summarize_text"]),
    ("Hello, how are you?", ["general_chat"]),
    ("", ["general_chat"]),
]


def run_tests():
    print("=== Intent Tests ===")
    passed = 0

    with patch("src.intent.client.chat.completions.create") as mock_create:
        for text, expected in TEST_CASES:
            # Prepare mock response
            mock_resp = MagicMock()
            if not text.strip():
                # The code handles empty text before the API call
                results = classify_compound_intent(text)
            else:
                intent_val = expected[0]
                mock_resp.choices[0].message.content = json.dumps(
                    {
                        "intents": [
                            {
                                "intent": intent_val,
                                "details": "mocked",
                                "filename": "mock.txt",
                            }
                        ]
                    }
                )
                mock_create.return_value = mock_resp
                results = classify_compound_intent(text)

            intents = [r["intent"] for r in results]
            print(f"Input: {text or '(empty)'} -> Intents: {intents}")

            if any(i in expected for i in intents):
                passed += 1
            else:
                print(f"FAILED: Expected {expected}, got {intents}")

    print(f"\nResults: {passed}/{len(TEST_CASES)} passed")
    assert passed == len(TEST_CASES)


if __name__ == "__main__":
    run_tests()
