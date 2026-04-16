import json
from groq import Groq
from src.config import GROQ_API_KEY, INTENT_MODEL

client = Groq(api_key=GROQ_API_KEY)

SUPPORTED_INTENTS = ["create_file", "write_code", "summarize_text", "general_chat"]

SYSTEM_PROMPT = """
You are an intent classifier for a voice-controlled AI.
Analyze the user command and return a JSON LIST of objects.
Detection should support multiple commands in one sentence (compound intents).

Intents:
- create_file: create new file/folder
- write_code: write or save code
- summarize_text: summarize content
- general_chat: questions, conversation

Rules:
- ALWAYS return a list of JSON objects: [{"intent": "...", "details": "...", "filename": "..."}]
- Suggested filename must have a relevant extension.
- Details should be concise.

Example Input: "Create a notes file and summarize the weather article"
Example Output: [
  {"intent": "create_file", "details": "notes file", "filename": "notes.txt"},
  {"intent": "summarize_text", "details": "weather article", "filename": "summary.txt"}
]
"""


def classify_compound_intent(text: str) -> list[dict]:
    """Classify user command into a list of intents (supporting multiple commands)."""
    if not text.strip():
        return [
            {
                "intent": "general_chat",
                "details": "Empty input",
                "filename": "output.txt",
            }
        ]

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

        # Groq's json_object mode requires a root key or it might just return the list inside an object.
        # We'll handle both cases.
        raw_data = json.loads(resp.choices[0].message.content)

        # If the model wraps it in an object like {"intents": [...]}, extract it.
        if isinstance(raw_data, dict):
            if "intents" in raw_data:
                intents = raw_data["intents"]
            elif "intent" in raw_data:  # Single object case
                intents = [raw_data]
            else:  # Fallback to a single generic intent if format is weird
                intents = [raw_data]
        else:
            intents = raw_data if isinstance(raw_data, list) else [raw_data]

        # Final validation and cleanup
        validated = []
        for i in intents:
            if not isinstance(i, dict):
                continue
            if i.get("intent") not in SUPPORTED_INTENTS:
                i["intent"] = "general_chat"
            i.setdefault("details", "No details")
            i.setdefault("filename", "output.txt")
            validated.append(i)

        return (
            validated
            if validated
            else [{"intent": "general_chat", "details": text, "filename": "output.txt"}]
        )

    except Exception:
        return [{"intent": "general_chat", "details": text, "filename": "output.txt"}]
