import os
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def safe_path(filename: str) -> str:
    """Ensure all files are written inside output/ and strip path traversal attempts."""
    filename = os.path.basename(filename)
    filename = re.sub(r"[^\w\-_\.]", "_", filename)
    return os.path.join(OUTPUT_DIR, filename)


def extract_code(raw: str) -> str:
    """Strip markdown code fences from LLM response."""
    match = re.search(r"```(?:\w+)?\n(.*?)```", raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw.strip()


def create_file(filename: str, content: str = "") -> str:
    """Create a new file in output/ with optional content."""
    try:
        path = safe_path(filename)
        if os.path.exists(path):
            print(f"Overwriting existing file: {path}")

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        size = os.path.getsize(path)
        return f"File created: {path} ({size} bytes)"
    except Exception as e:
        return f"Failed to create file: {str(e)}"


def write_code(filename: str, details: str) -> tuple[str, str]:
    """Generate code using LLM and save to output/."""
    try:
        ext = os.path.splitext(filename)[1].lower()
        language_map = {
            ".py": "Python",
            ".js": "JavaScript",
            ".ts": "TypeScript",
            ".java": "Java",
            ".cpp": "C++",
            ".c": "C",
            ".go": "Go",
            ".rs": "Rust",
            ".html": "HTML",
            ".css": "CSS",
            ".sh": "Bash",
            ".sql": "SQL",
        }
        language = language_map.get(ext, "Python")

        prompt = f"Write clean, well-commented {language} code for: {details}. Return ONLY code."
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        clean_code = extract_code(response.choices[0].message.content)
        status = create_file(filename, clean_code)
        return status, clean_code
    except Exception as e:
        return f"Code generation failed: {str(e)}", ""


def summarize_text(
    text: str, save_to_file: bool = False, filename: str = "summary.txt"
) -> str:
    """Summarize text using LLM, optionally saving to a file."""
    try:
        if not text.strip():
            return "No text provided to summarize."

        prompt = f"Summarize the following text clearly and concisely:\n{text}"
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        summary = response.choices[0].message.content.strip()
        if save_to_file:
            create_file(filename, summary)
            summary += f"\n\nSummary saved to output/{filename}"
        return summary
    except Exception as e:
        return f"Summarization failed: {str(e)}"


def general_chat(text: str) -> str:
    """Handle general questions and conversation."""
    try:
        if not text.strip():
            return "No input provided."

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant. Give clear, concise answers.",
                },
                {"role": "user", "content": text},
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Chat failed: {str(e)}"


def create_folder(foldername: str) -> str:
    """Create a subfolder inside output/."""
    try:
        foldername = re.sub(r"[^\w\-_]", "_", foldername)
        path = os.path.join(OUTPUT_DIR, foldername)
        os.makedirs(path, exist_ok=True)
        return f"Folder created: {path}"
    except Exception as e:
        return f"Failed to create folder: {str(e)}"


def execute_tool(intent_data: dict) -> str:
    """Dispatcher to call the appropriate tool based on intent data."""
    intent = intent_data.get("intent", "general_chat")
    details = intent_data.get("details", "")
    filename = intent_data.get("filename", "output.txt")

    if intent == "create_file":
        return create_file(filename)
    elif intent == "write_code":
        status, _ = write_code(filename, details)
        return status
    elif intent == "summarize_text":
        save = any(word in details.lower() for word in ["save", "write"])
        return summarize_text(details, save_to_file=save, filename=filename)
    elif intent == "general_chat":
        return general_chat(details)
    return f"Unknown intent: {intent}"
