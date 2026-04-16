import json
from groq import Groq
from src.config import GROQ_API_KEY, INTENT_MODEL

client = Groq(api_key=GROQ_API_KEY)

SUPPORTED_INTENTS = ["create_file", "write_code", "summarize_text", "general_chat"]

SYSTEM_PROMPT = """
You are an intent classifier for a voice-controlled AI.
Parse the user command and return ONLY a JSON object.

Intents:
- create_file: create new file/folder
- write_code: write or save code
- summarize_text: summarize content
- general_chat: questions, conversation

Rules:
- For file + code, use write_code.
- Include a suggested filename with extension.
- Details should be concise.

Format:
{
  "intent": "intent_name",
  "details": "task details",
  "filename": "suggested_name.ext"
}
"""


def classify_intent(text: str) -> dict:
    if not text.strip():
        return {
            "intent": "general_chat",
            "details": "Empty input",
            "filename": "output.txt",
            "error": "Empty input",
        }

    try:
        resp = client.chat.completions.create(
            model=INTENT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )

        data = json.loads(resp.choices[0].message.content)

        # Simple validation
        if data.get("intent") not in SUPPORTED_INTENTS:
            data["intent"] = "general_chat"

        data.setdefault("details", "No details")
        data.setdefault("filename", "output.txt")
        data["error"] = None

        return data

    except Exception as e:
        return {
            "intent": "general_chat",
            "details": text,
            "filename": "output.txt",
            "error": str(e),
        }
