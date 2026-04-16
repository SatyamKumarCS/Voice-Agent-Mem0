import os
import sys
from pathlib import Path
from typing import TypedDict
import gradio as gr
from langgraph.graph import StateGraph, END

sys.path.append(str(Path(__file__).parent.parent))

from src.stt import transcribe_from_gradio
from src.intent import classify_compound_intent
from src.tools import execute_tool
from src.config import SERVER_NAME, SERVER_PORT


class AgentState(TypedDict):
    audio_path: str
    transcription: str
    intents: list[dict]
    results: list[str]


def transcribe_node(state: AgentState) -> AgentState:
    result = transcribe_from_gradio(state["audio_path"])
    if not result["success"]:
        raise RuntimeError(result["error"])
    return {**state, "transcription": result["text"]}


def intent_node(state: AgentState) -> AgentState:
    return {**state, "intents": classify_compound_intent(state["transcription"])}


def tool_node(state: AgentState) -> AgentState:
    return {**state, "results": [execute_tool(i) for i in state["intents"]]}


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("transcribe_node", transcribe_node)
    g.add_node("intent_node", intent_node)
    g.add_node("tool_node", tool_node)
    g.set_entry_point("transcribe_node")
    g.add_edge("transcribe_node", "intent_node")
    g.add_edge("intent_node", "tool_node")
    g.add_edge("tool_node", END)
    return g.compile()


agent = build_graph()

INTENT_LABELS = {
    "create_file": "📄 Create File",
    "write_code": "💻 Write Code",
    "summarize_text": "📝 Summarize",
    "general_chat": "💬 Chat",
}

INTENT_COLORS = {
    "create_file": "#e0f2fe",
    "write_code": "#ede9fe",
    "summarize_text": "#fef3c7",
    "general_chat": "#dcfce7",
}

INTENT_BORDERS = {
    "create_file": "#7dd3fc",
    "write_code": "#a78bfa",
    "summarize_text": "#fbbf24",
    "general_chat": "#86efac",
}


def run_pipeline(audio_input):
    if audio_input is None:
        return "_Waiting for a command…_", "", "", format_output_files()
    try:
        state = agent.invoke(
            {
                "audio_path": audio_input,
                "transcription": "",
                "intents": [],
                "results": [],
            }
        )
        intents = state["intents"]
        results = state["results"]
        return (
            f"> {state['transcription']}",
            format_intents(intents),
            format_results(results),
            format_output_files(),
        )
    except Exception as e:
        return "—", "", f"**Error:** {str(e)}", format_output_files()


def format_intents(intents: list) -> str:
    if not intents:
        return ""
    parts = []
    for d in intents:
        intent = d.get("intent", "")
        label = INTENT_LABELS.get(intent, "⚡ Action")
        bg = INTENT_COLORS.get(intent, "#f1f5f9")
        border = INTENT_BORDERS.get(intent, "#cbd5e1")
        fname = d.get("filename", "")
        detail = d.get("details", "")
        file_html = (
            f"<code style='font-size:0.78rem;background:#f8fafc;padding:2px 7px;border-radius:4px;color:#334155'>{fname}</code>"
            if fname and fname not in ("output.txt", "")
            else ""
        )
        parts.append(
            f"<div style='background:{bg};border:1px solid {border};border-radius:10px;"
            f"padding:12px 16px;margin-bottom:8px'>"
            f"<div style='font-weight:600;font-size:0.88rem;color:#1e293b;margin-bottom:4px'>{label}</div>"
            f"{file_html}"
            f"<div style='font-size:0.82rem;color:#475569;margin-top:4px'>{detail}</div>"
            f"</div>"
        )
    return "".join(parts)


def format_results(results: list) -> str:
    if not results:
        return ""
    return "\n\n---\n\n".join(results)


def format_output_files() -> str:
    output_dir = "output"
    if not os.path.exists(output_dir):
        return "<p style='color:#94a3b8;font-size:0.85rem'>No files yet.</p>"
    files = sorted(
        [
            f
            for f in os.listdir(output_dir)
            if os.path.isfile(os.path.join(output_dir, f))
        ]
    )
    if not files:
        return "<p style='color:#94a3b8;font-size:0.85rem'>No files yet.</p>"
    rows = []
    for f in files:
        size = os.path.getsize(os.path.join(output_dir, f))
        rows.append(
            f"<div style='display:flex;justify-content:space-between;align-items:center;"
            f"padding:8px 12px;border-radius:8px;background:#f8fafc;margin-bottom:6px;"
            f"border:1px solid #e2e8f0'>"
            f"<span style='font-size:0.84rem;font-weight:500;color:#334155'>📄 {f}</span>"
            f"<span style='font-size:0.75rem;color:#94a3b8'>{size} B</span>"
            f"</div>"
        )
    return "".join(rows)


def view_file(filename: str) -> str:
    if not filename:
        return ""
    path = os.path.join("output", filename.strip())
    if not os.path.exists(path):
        return f"File not found: {filename}"
    try:
        return open(path, encoding="utf-8").read()
    except Exception as e:
        return f"Cannot read file: {e}"


# ── CSS ──────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── Global ── */
html, body {
    background: #f8fafc !important;
}

.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: #f8fafc !important;
    max-width: 900px !important;
    margin: 0 auto !important;
    padding: 0 24px 60px !important;
}

footer { display: none !important; }
.block { box-shadow: none !important; }

/* ── Header ── */
.va-header {
    text-align: center;
    padding: 48px 0 32px;
}
.va-header h1 {
    font-size: 1.75rem;
    font-weight: 700;
    color: #0f172a;
    margin: 0 0 6px;
    letter-spacing: -0.03em;
}
.va-header p {
    font-size: 0.88rem;
    color: #64748b;
    margin: 0;
    font-weight: 400;
}
.va-header .va-badge {
    display: inline-block;
    margin-top: 12px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    color: #6366f1;
    background: #eef2ff;
    padding: 4px 12px;
    border-radius: 100px;
}

/* ── Section Labels ── */
.va-section {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 10px;
    display: block;
}

/* ── Cards ── */
.va-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 14px;
    transition: box-shadow 0.2s ease;
}
.va-card:hover {
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
}

/* ── Audio widget ── */
#input-area .block,
#input-area audio {
    border-radius: 10px !important;
    border-color: #e2e8f0 !important;
}

/* ── Primary button ── */
#run-btn {
    background: #6366f1 !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    color: #fff !important;
    height: 42px !important;
    box-shadow: 0 1px 3px rgba(99,102,241,0.25) !important;
    transition: all 0.15s ease !important;
}
#run-btn:hover {
    background: #4f46e5 !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.3) !important;
    transform: translateY(-1px) !important;
}
#run-btn:active {
    transform: translateY(0) !important;
}

/* ── Secondary button ── */
#clear-btn {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    color: #64748b !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    height: 42px !important;
    transition: all 0.15s !important;
}
#clear-btn:hover {
    border-color: #cbd5e1 !important;
    background: #f1f5f9 !important;
    color: #334155 !important;
}

/* ── Transcript blockquote ── */
#transcript-area .prose blockquote {
    border-left: 3px solid #6366f1;
    padding-left: 14px;
    margin: 0;
    color: #334155;
    font-size: 0.95rem;
}
#transcript-area .prose em {
    color: #94a3b8;
}

/* ── Output markdown ── */
#output-area .prose {
    color: #334155 !important;
    font-size: 0.9rem;
    line-height: 1.7;
}
#output-area .prose code {
    background: #f1f5f9;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.82rem;
}

/* ── Accordion ── */
#files-accordion > .label-wrap {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    color: #64748b !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 12px 18px !important;
}
#files-accordion[open] > .label-wrap {
    border-bottom-left-radius: 0 !important;
    border-bottom-right-radius: 0 !important;
    border-bottom-color: transparent !important;
}
#files-accordion > .inner {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-top: none !important;
    border-bottom-left-radius: 12px !important;
    border-bottom-right-radius: 12px !important;
    padding: 16px 20px !important;
}

/* ── Textbox & Code viewer ── */
#file-input input {
    border-radius: 8px !important;
    border-color: #e2e8f0 !important;
    font-size: 0.85rem !important;
}
#view-btn {
    border-radius: 8px !important;
    font-size: 0.8rem !important;
}
#file-viewer {
    border-radius: 10px !important;
    border-color: #e2e8f0 !important;
}

/* ── Status dot animation ── */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    background: #22c55e;
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 2s ease-in-out infinite;
}
"""


def build_ui():
    with gr.Blocks(title="Voice Agent") as demo:
        # ── Header ──
        gr.HTML("""
        <div class="va-header">
            <h1>🎙️ Voice Agent</h1>
            <p>Speak a command — the agent transcribes, understands, and acts.</p>
            <span class="va-badge"><span class="status-dot"></span>POWERED BY LANGGRAPH + GROQ</span>
        </div>
        """)

        # ── Input Card ──
        gr.HTML("<span class='va-section'>Command</span>")
        with gr.Group(elem_id="input-area"):
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                show_label=False,
            )
            with gr.Row():
                run_btn = gr.Button(
                    "Run Agent ▸", variant="primary", elem_id="run-btn", scale=5
                )
                clear_btn = gr.Button("Clear", elem_id="clear-btn", scale=1)

        # ── Transcription ──
        gr.HTML("<span class='va-section'>Transcription</span>")
        with gr.Group(elem_id="transcript-area"):
            transcription_box = gr.Markdown(value="_Waiting for a command…_")

        # ── Actions & Output side by side ──
        with gr.Row(equal_height=True):
            with gr.Column(scale=2):
                gr.HTML("<span class='va-section'>Detected Actions</span>")
                with gr.Group():
                    intent_box = gr.HTML(
                        value="<p style='color:#94a3b8;font-size:0.85rem'>Actions will appear here.</p>"
                    )
            with gr.Column(scale=3):
                gr.HTML("<span class='va-section'>Output</span>")
                with gr.Group(elem_id="output-area"):
                    result_box = gr.Markdown(value="_Results will appear here._")

        # ── File Explorer ──
        with gr.Accordion("📁  Output Files", open=False, elem_id="files-accordion"):
            with gr.Row():
                with gr.Column(scale=1):
                    files_display = gr.HTML(value=format_output_files())
                    refresh_btn = gr.Button("↻ Refresh", size="sm")
                with gr.Column(scale=2):
                    file_input = gr.Textbox(
                        placeholder="e.g. retry.py",
                        label="View a file",
                        elem_id="file-input",
                    )
                    view_btn = gr.Button("View", size="sm", elem_id="view-btn")
                    file_output = gr.Code(
                        show_label=False,
                        interactive=False,
                        lines=14,
                        elem_id="file-viewer",
                    )

        # ── Events ──
        run_btn.click(
            fn=run_pipeline,
            inputs=audio_input,
            outputs=[transcription_box, intent_box, result_box, files_display],
        )
        clear_btn.click(
            fn=lambda: (
                "_Waiting for a command…_",
                "<p style='color:#94a3b8;font-size:0.85rem'>Actions will appear here.</p>",
                "_Results will appear here._",
                format_output_files(),
            ),
            outputs=[transcription_box, intent_box, result_box, files_display],
        )
        refresh_btn.click(fn=format_output_files, outputs=files_display)
        view_btn.click(fn=view_file, inputs=file_input, outputs=file_output)

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        server_name=SERVER_NAME,
        server_port=SERVER_PORT,
        css=CSS,
    )
