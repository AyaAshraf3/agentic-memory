"""
Live demo: a personalized learning assistant with LangGraph memory.

  * Short-term memory  -> checkpointer, keyed by thread_id
  * Long-term memory   -> store, keyed by ("learners", user_id, "profile")

Run the five scripted tests:          python memory_demo.py
Chat interactively:                   python memory_demo.py --chat --user u_maya --thread t1
Browser UI:                           streamlit run app.py
Use the Sprints LiteLLM proxy:        set LITELLM_BASE_URL, LITELLM_API_KEY, LITELLM_MODEL (see .env.example)
Or any LangChain model string:        set MODEL=anthropic:claude-sonnet-5  (or openai:gpt-4o-mini)

Without a model configured, a small rule-based "offline tutor" is used, so the
demo never depends on Wi-Fi or API keys. The memory mechanics are identical.
"""

import argparse
import json
import os
import re
from dataclasses import dataclass

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore


@dataclass
class Context:
    user_id: str


PROFILE_KEY = "profile"


def profile_ns(user_id: str) -> tuple:
    # Built in code from the trusted user_id, never from model output.
    return ("learners", user_id, "profile")


# ---------------------------------------------------------------------------
# Model: a real chat model when MODEL is set, otherwise an offline stand-in
# ---------------------------------------------------------------------------

try:
    from dotenv import load_dotenv

    load_dotenv()  # picks up LITELLM_* / MODEL from a local .env file
except ImportError:
    pass


def make_llm(model: str | None = None):
    """LiteLLM proxy if LITELLM_BASE_URL is set, else a MODEL string, else None (offline tutor)."""
    base_url = os.getenv("LITELLM_BASE_URL")
    if base_url:
        from langchain_openai import ChatOpenAI  # LiteLLM speaks the OpenAI API

        return ChatOpenAI(
            base_url=base_url.rstrip("/"),
            api_key=os.getenv("LITELLM_API_KEY"),
            model=model or os.getenv("LITELLM_MODEL", "gpt-4o-mini"),
            temperature=0,
        )
    if model or os.getenv("MODEL"):
        from langchain.chat_models import init_chat_model

        return init_chat_model(model or os.getenv("MODEL"))
    return None


def list_litellm_models() -> list[str]:
    """Model names the LiteLLM proxy exposes (empty list if unreachable)."""
    base_url = os.getenv("LITELLM_BASE_URL")
    if not base_url:
        return []
    from openai import OpenAI

    client = OpenAI(base_url=base_url.rstrip("/"), api_key=os.getenv("LITELLM_API_KEY"))
    return sorted(m.id for m in client.models.list())


llm = make_llm()
MODEL = getattr(llm, "model_name", None) or os.getenv("MODEL")


def offline_reply(profile: dict, messages: list) -> str:
    question = messages[-1].content.lower()
    role = profile.get("role")
    if "what's my job" in question or "what is my role" in question or "what's my role" in question:
        return f"You told me you're a {role}." if role else "I don't know your role yet."
    if "what did i ask first" in question:
        first = next((m.content for m in messages if isinstance(m, HumanMessage)), None)
        return f'Your first message was: "{first}"'
    if "maya" in question and "role" in question:
        return "I can only see your own learning profile."
    if "group by" in question:
        if role == "data analyst":
            return "GROUP BY collapses rows per key - e.g. total sales per region in your sales table."
        if role == "teacher":
            return "GROUP BY buckets rows by a column - like grouping student grades by class."
        return "GROUP BY groups rows that share a value so you can aggregate them. What's your background?"
    return "Got it!"


def offline_extract(profile: dict, text: str) -> dict:
    text = text.lower()
    changes = {}
    if m := re.search(r"i(?:'m| am) an? ([a-z ]+?)(?:[.,!]|$)", text):
        changes["role"] = m.group(1).strip()
    if "i teach" in text:
        changes["role"] = "teacher"
    if m := re.search(r"i (?:prefer|like) ([a-z ,]+?)(?:[.!]|$)", text):
        changes["preferences"] = m.group(1).strip()
    if "forget my preferences" in text:
        changes["preferences"] = None  # None = delete this field
    return changes


def llm_extract(profile: dict, text: str) -> dict:
    prompt = (
        "You maintain a learner profile with keys role, level, preferences.\n"
        f"Current profile: {json.dumps(profile)}\n"
        f"Latest learner message: {text!r}\n"
        "Return ONLY a JSON object with keys that are new or corrected, stable facts. "
        "Use null for a key the learner asked you to forget. Return {} if nothing changed."
    )
    raw = llm.invoke(prompt).content
    match = re.search(r"\{.*\}", raw, re.S)
    return json.loads(match.group(0)) if match else {}


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def tutor(state: MessagesState, runtime: Runtime[Context]):
    item = runtime.store.get(profile_ns(runtime.context.user_id), PROFILE_KEY)
    profile = item.value if item else {}
    if llm:
        system = SystemMessage(f"You are a patient tutor. Learner profile: {json.dumps(profile)}. Keep answers short.")
        reply = llm.invoke([system] + state["messages"])
    else:
        reply = AIMessage(offline_reply(profile, state["messages"]))
    return {"messages": [reply]}


def save_memories(state: MessagesState, runtime: Runtime[Context]):
    ns = profile_ns(runtime.context.user_id)
    item = runtime.store.get(ns, PROFILE_KEY)
    profile = item.value if item else {}
    last_user = next(m for m in reversed(state["messages"]) if isinstance(m, HumanMessage))
    changes = (llm_extract if llm else offline_extract)(profile, last_user.content)
    if changes:
        merged = {**profile, **changes}
        merged = {k: v for k, v in merged.items() if v is not None}
        runtime.store.put(ns, PROFILE_KEY, merged)
    return {}


def build_graph():
    builder = StateGraph(MessagesState, context_schema=Context)
    builder.add_node("tutor", tutor)
    builder.add_node("save_memories", save_memories)
    builder.add_edge(START, "tutor")
    builder.add_edge("tutor", "save_memories")
    builder.add_edge("save_memories", END)
    store = InMemoryStore()
    graph = builder.compile(checkpointer=InMemorySaver(), store=store)
    return graph, store


# ---------------------------------------------------------------------------
# Demo runner
# ---------------------------------------------------------------------------

def say(graph, user_id, thread_id, text):
    out = graph.invoke(
        {"messages": [HumanMessage(text)]},
        {"configurable": {"thread_id": f"{user_id}:{thread_id}"}},
        context=Context(user_id=user_id),
    )
    answer = out["messages"][-1].content
    print(f"  [{user_id} / {thread_id}] you: {text}")
    print(f"  [{user_id} / {thread_id}] bot: {answer}")
    return answer


def show_store(store, user_id):
    item = store.get(profile_ns(user_id), PROFILE_KEY)
    print(f"  store {profile_ns(user_id)} -> {item.value if item else None}")


def run_tests():
    graph, store = build_graph()
    name = getattr(llm, "model_name", None) or getattr(llm, "model", None)
    print(f"\nModel: {name or 'offline tutor (configure LiteLLM or MODEL to use a real LLM)'}")

    print("\n1) Same conversation - short-term memory")
    say(graph, "u_maya", "t1", "Hi, I'm a data analyst.")
    say(graph, "u_maya", "t1", "What did I ask first?")
    msgs = graph.get_state({"configurable": {"thread_id": "u_maya:t1"}}).values["messages"]
    print(f"  checkpointer holds {len(msgs)} messages for thread t1")

    print("\n2) New conversation - long-term memory")
    say(graph, "u_maya", "t2", "Explain GROUP BY.")
    show_store(store, "u_maya")

    print("\n3) Correction - update, don't duplicate")
    say(graph, "u_maya", "t3", "Actually, I teach now. I'm a teacher.")
    say(graph, "u_maya", "t4", "Explain GROUP BY.")
    show_store(store, "u_maya")

    print("\n4) Isolation - another learner")
    say(graph, "u_omar", "t1", "What's my role? And what's Maya's role?")
    show_store(store, "u_omar")

    print("\n5) Forget on request")
    say(graph, "u_maya", "t5", "I prefer short answers.")
    show_store(store, "u_maya")
    say(graph, "u_maya", "t5", "Please forget my preferences.")
    show_store(store, "u_maya")

    print("\nEverything in the store:")
    for item in store.search(("learners",)):
        print(f"  {item.namespace} / {item.key} -> {item.value}")


def chat(user_id, thread_id):
    graph, store = build_graph()
    print(f"Chatting as {user_id} on thread {thread_id}. Commands: /store, /thread <id>, /user <id>, /quit")
    while True:
        text = input("> ").strip()
        if text == "/quit":
            break
        if text == "/store":
            show_store(store, user_id)
            continue
        if text.startswith("/thread "):
            thread_id = text.split(maxsplit=1)[1]
            continue
        if text.startswith("/user "):
            user_id = text.split(maxsplit=1)[1]
            continue
        say(graph, user_id, thread_id, text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--user", default="u_maya")
    parser.add_argument("--thread", default="t1")
    args = parser.parse_args()
    chat(args.user, args.thread) if args.chat else run_tests()
