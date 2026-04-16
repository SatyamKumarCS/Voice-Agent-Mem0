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

    for text, expected in TEST_CASES:
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
