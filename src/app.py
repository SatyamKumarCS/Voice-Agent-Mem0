from typing import TypedDict
from langgraph.graph import StateGraph, END
from src.stt import transcribe_text
from src.intent import classify_intent
from src.tools import create_file, write_code, summarize_text, general_chat


class AgentState(TypedDict):
    audio_path: str
    transcription: str
    intent: str
    details: str
    filename: str
    result: str


def transcribe_node(state: AgentState) -> AgentState:
    text = transcribe_text(state["audio_path"])
    return {**state, "transcription": text}


def intent_node(state: AgentState) -> AgentState:
    data = classify_intent(state["transcription"])
    return {
        **state,
        "intent": data["intent"],
        "details": data["details"],
        "filename": data["filename"],
    }


def tool_router(state: AgentState) -> str:
    mapping = {
        "create_file": "create_file_node",
        "write_code": "write_code_node",
        "summarize_text": "summarize_node",
        "general_chat": "chat_node",
    }
    return mapping.get(state["intent"], "chat_node")


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("transcribe_node", transcribe_node)
    workflow.add_node("intent_node", intent_node)
    workflow.add_node(
        "create_file_node", lambda s: {**s, "result": create_file(s["filename"])}
    )
    workflow.add_node(
        "write_code_node",
        lambda s: {**s, "result": write_code(s["filename"], s["details"])[0]},
    )
    workflow.add_node(
        "summarize_node", lambda s: {**s, "result": summarize_text(s["details"])}
    )
    workflow.add_node(
        "chat_node", lambda s: {**s, "result": general_chat(s["details"])}
    )

    workflow.set_entry_point("transcribe_node")
    workflow.add_edge("transcribe_node", "intent_node")

    workflow.add_conditional_edges(
        "intent_node",
        tool_router,
        {
            k: k
            for k in [
                "create_file_node",
                "write_code_node",
                "summarize_node",
                "chat_node",
            ]
        },
    )

    for node in ["create_file_node", "write_code_node", "summarize_node", "chat_node"]:
        workflow.add_edge(node, END)

    return workflow.compile()


def run_agent(audio_path: str):
    agent = build_graph()
    return agent.invoke({"audio_path": audio_path})


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        res = run_agent(sys.argv[1])
        print(f"\nFinal State:\n{res}")
    else:
        print("Usage: python src/app.py <audio_path>")
