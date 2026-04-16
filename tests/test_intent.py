from src.intent import classify_intent

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
        res = classify_intent(text)
        print(f"Input: {text or '(empty)'} -> Intent: {res['intent']}")
        
        if res["intent"] in expected:
            passed += 1
        else:
            print(f"FAILED: Expected {expected}, got {res['intent']}")
            
    print(f"\nResults: {passed}/{len(TEST_CASES)} passed")
    assert passed == len(TEST_CASES)

if __name__ == "__main__":
    run_tests()
