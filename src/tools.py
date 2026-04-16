from groq import Groq
from src.config import GROQ_API_KEY, LLM_MODEL, OUTPUT_DIR

client = Groq(api_key=GROQ_API_KEY)


def create_file(filename: str, content: str = "") -> str:
    path = OUTPUT_DIR / filename
    path.write_text(content)
    return f"Created: {path}"


def write_code(filename: str, details: str) -> tuple[str, str]:
    prompt = f"Write clean Python code for: {details}. Return only the code."
    resp = client.chat.completions.create(
        model=LLM_MODEL, messages=[{"role": "user", "content": prompt}]
    )
    code = resp.choices[0].message.content
    return create_file(filename, code), code


def summarize_text(text: str) -> str:
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": f"Summarize in 3 sentences: {text}"}],
    )
    return resp.choices[0].message.content


def general_chat(text: str) -> str:
    resp = client.chat.completions.create(
        model=LLM_MODEL, messages=[{"role": "user", "content": text}]
    )
    return resp.choices[0].message.content
