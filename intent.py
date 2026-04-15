import ollama
import json

def classify_intent(text: str) -> dict:
    prompt = f"""
You are an intent classifier. Given a user command, return ONLY a JSON object
Possible intents: create_file, write_code, summarize_text, general_chat
User command: "{text}"
Respond ONLY with JSON like:
{{"intent": "write_code", "details": "retry function in Python", "filename": "retry.py"}}
"""
    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response['message']['content']
    return json.loads(raw)
