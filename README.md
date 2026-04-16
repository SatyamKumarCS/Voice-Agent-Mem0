# 🎙️ Voice Agent Mem0

A professional-grade voice assistant powered by **LangGraph** and **Groq**. This agent transcribes speech, identifies compound intents, and executes localized tools in a sleek, light-themed Gradio interface.

## ✨ Features
- **Compound Intent Processing**: Understands multiple commands in a single voice recording.
- **Tool Integration**: Automatically creates files, writes code, and summarizes text.
- **Groq-Powered**: Lightning-fast inference using `whisper-large-v3` for STT and `llama-3.3-70b` for reasoning.
- **Clean UI**: Minimalist, responsive Gradio dashboard with output file management.

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have Python 3.10+ and a [Groq API Key](https://console.groq.com/).

### 2. Setup
```bash
# Clone the repository
git clone https://github.com/SatyamKumarCS/Voice-Agent-Mem0.git
cd Voice-Agent-Mem0

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_api_key_here
SERVER_NAME=0.0.0.0
SERVER_PORT=7860
```

### 4. Run
```bash
python -m src.app
```
Access the UI at `http://localhost:7860`.

## 📁 Project Structure
- `src/app.py`: Main entry point & Gradio UI logic.
- `src/intent.py`: Intent classification engine.
- `src/tools.py`: Tool execution logic (File creation, Code gen, etc).
- `src/stt.py`: Integration with Groq Whisper API.
- `output/`: Generated files directory.

## 🧪 Testing
Run the test suite using `pytest`:
```bash
pytest tests/
```
