# 🎙️ Voice Agent Mem0

A professional-grade voice assistant powered by **LangGraph** and **Groq**. This agent transcribes speech, identifies compound intents, and executes localized tools in a sleek, modern Gradio interface.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Stateful%20AI-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Groq](https://img.shields.io/badge/Groq-Ultra--Fast%20Inference-green.svg)](https://groq.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## ✨ Features
- **🧠 Compound Intent Processing**: Handles multiple commands in a single voice recording (e.g., "Summarize this and save it to a file").
- **🛠️ Automated Toolset**: Built-in capabilities for file creation, code generation, and intelligent text summarization.
- **⚡ Groq-Powered Performance**: Sub-second speech transcription using `whisper-large-v3` and reasoning via `llama-3.3-70b`.
- **🎨 Premium UI**: A responsive, light-themed Gradio dashboard with a built-in file explorer and micro-animations.

---

## 🌐 Deployment (Render)

Deploying to Render is seamless thanks to the included Blueprint configuration.

### 1. The Blueprint Way (Recommended)
1. Fork this repository.
2. Go to [Render Blueprints](https://dashboard.render.com/blueprints).
3. Connect your fork.
4. Set your `GROQ_API_KEY` when prompted.
5. Click **Apply**.

### 2. Manual Web Service Setup
- **Environment**: `Python 3`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python -m src.app`
- **Port**: Render handles this automatically.

---

## 🚀 Local Installation

### 1. Prerequisites
- Python 3.10 or higher
- A [Groq API Key](https://console.groq.com/)

### 2. Setup & Install
```bash
# Clone and enter the repo
git clone https://github.com/SatyamKumarCS/Voice-Agent-Mem0.git
cd Voice-Agent-Mem0

# Initialize virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. Configuration
Rename `.env.example` to `.env` and add your key:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Run the Agent
```bash
python -m src.app
```
Then open your browser to `http://localhost:7860`.

---

## 📁 Project Architecture
```text
├── src/
│   ├── app.py          # Main application & UI
│   ├── intent.py       # Intent classification engine
│   ├── tools.py        # Logic for writing code, files, etc.
│   ├── stt.py          # Groq Whisper STT integration
│   └── config.py       # Global settings & environment
├── tests/              # Intent and STT validation suite
├── output/             # Directory for generated files
└── render.yaml         # Deployment configuration
```

---

## 🧪 Testing
We maintain high reliability with a focused test suite:
```bash
pytest tests/
```

---

## 📄 License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
