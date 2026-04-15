import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client=Groq(api_key=os.getenv("GROQ_API_KEY"))

def transcribe(audio_path: str) -> str:
    with open(audio_path, "rb") as f:
        result= client.audio .transcriptions.create(
            model="whisper-large-v3",
            file=f,
            response_format="text"
        ) 
    return result