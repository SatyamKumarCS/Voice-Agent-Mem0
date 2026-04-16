import os
from unittest.mock import patch, MagicMock
from src.tools import (
    create_file,
    write_code,
    summarize_text,
    general_chat,
    create_folder,
    execute_tool,
)

passed = 0
failed = 0


def test(label, condition, extra=""):
    global passed, failed
    if condition:
        print(f"  PASS  {label}")
        passed += 1
    else:
        print(f"  FAIL  {label} {extra}")
        failed += 1


def run_tests():
    global passed, failed
    # Ensure output dir exists
    os.makedirs("output", exist_ok=True)

    with patch("src.tools.client.chat.completions.create") as mock_create:
        # 1. Mock response for LLM calls
        mock_resp = MagicMock()
        mock_resp.choices[
            0
        ].message.content = "Mocked LLM Response content that is long enough to pass the length checks in the tests."
        mock_create.return_value = mock_resp

        print("\n=== create_file ===")
        result = create_file("hello.txt", "Hello World!")
        test("File created successfully", "created" in result.lower())
        test("File exists on disk", os.path.exists("output/hello.txt"))
        test(
            "File has correct content",
            open("output/hello.txt").read() == "Hello World!",
        )

        result = create_file("../../etc/passwd", "hack")
        test(
            "Path traversal blocked",
            "etc" not in result or os.path.exists("output/passwd"),
        )

        print("\n=== write_code ===")
        status, code = write_code("retry.py", "retry function with exponential backoff")
        test("Code generated", len(code) > 50)
        test("Code file created", os.path.exists("output/retry.py"))
        test("Code file not empty", os.path.getsize("output/retry.py") > 0)
        test("Status reports success", "created" in status.lower())

        status, code = write_code("sort.js", "function to sort an array of numbers")
        test("JS file created", os.path.exists("output/sort.js"))
        test("JS code generated", len(code) > 20)

        print("\n=== summarize_text ===")
        long_text = "Artificial intelligence is transforming every industry. Machine learning can detect diseases. NLP understands speech."
        summary = summarize_text(long_text)
        test("Summary returned", len(summary) > 20)
        test(
            "Summary contains logic",
            "failure" not in summary.lower() and "error" not in summary.lower(),
        )

        result = summarize_text("")
        test("Empty input handled", "no text" in result.lower())

        print("\n=== general_chat ===")
        response = general_chat("What is Python used for?")
        test("Chat responded", len(response) > 20)
        test("No error in response", "failed" not in response.lower())

        response = general_chat("")
        test("Empty chat handled", "no input" in response.lower())

        print("\n=== create_folder ===")
        result = create_folder("my_project")
        test("Folder created", os.path.exists("output/my_project"))
        test("Success message returned", "created" in result.lower())

        print("\n=== execute_tool dispatcher ===")
        # For execute_tool, we need to mock again if it calls LLM functions
        r = execute_tool(
            {"intent": "create_file", "details": "", "filename": "test.txt"}
        )
        test("Dispatcher: create_file", "created" in r.lower())

        r = execute_tool(
            {"intent": "write_code", "details": "hello world", "filename": "hello.py"}
        )
        test("Dispatcher: write_code", "created" in r.lower())

        r = execute_tool({"intent": "unknown_intent", "details": "", "filename": ""})
        test("Dispatcher: unknown intent", "unknown" in r.lower())

    print(f"\n=== Results: {passed} passed, {failed} failed ===\n")
    assert failed == 0


if __name__ == "__main__":
    run_tests()
