"""
Streamlit UI for the agent-memory demo.

    streamlit run app.py

Left: chat with the tutor as any learner, on any thread.
Right: what the agent remembers - the thread (checkpointer) and the learner profile (store).
"""

import contextlib
import io
import json

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

import memory_demo as md

st.set_page_config(page_title="Agent Memory Demo", page_icon="🧠", layout="wide")

OFFLINE = "Offline tutor (no API)"


@st.cache_resource
def get_graph():
    # One graph + store for the whole app, so memory survives reruns and new threads.
    return md.build_graph()


@st.cache_data(ttl=300, show_spinner=False)
def available_models():
    try:
        return md.list_litellm_models()
    except Exception:
        return []


graph, store = get_graph()

# ---------------------------------------------------------------------------
# Sidebar: model, learner, thread
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Setup")

    models = available_models()
    default_model = md.os.getenv("LITELLM_MODEL")
    options = [OFFLINE] + models
    if default_model and default_model not in options:
        options.insert(1, default_model)
    index = options.index(default_model) if default_model in options else (1 if len(options) > 1 else 0)
    choice = st.selectbox("Agent model (LiteLLM)", options, index=index)
    if not md.os.getenv("LITELLM_BASE_URL"):
        st.caption("LITELLM_BASE_URL is not set - only the offline tutor is available. See .env.example.")

    if st.session_state.get("model") != choice:
        md.llm = None if choice == OFFLINE else md.make_llm(choice)
        st.session_state.model = choice

    st.divider()
    user_id = st.selectbox("Learner (user_id)", ["u_maya", "u_omar", "u_sara"])
    st.session_state.setdefault("thread_n", {})
    n = st.session_state.thread_n.setdefault(user_id, 1)
    col_a, col_b = st.columns([2, 1])
    col_a.markdown(f"**Thread:** `t{n}`")
    if col_b.button("New", help="Start a new conversation (new thread_id)"):
        st.session_state.thread_n[user_id] = n + 1
        st.rerun()
    thread_id = f"t{st.session_state.thread_n[user_id]}"

    st.divider()
    st.markdown("**Flow**  \n`START → tutor → save_memories → END`")
    st.caption(
        "tutor reads the profile from the store and answers. "
        "save_memories extracts new facts and merges them into the store."
    )
    if st.button("Reset all memory", type="secondary"):
        get_graph.clear()
        st.session_state.clear()
        st.rerun()

config = {"configurable": {"thread_id": f"{user_id}:{thread_id}"}}
ns = md.profile_ns(user_id)

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("🧠 Personalized learning assistant")
st.caption(f"Chatting as **{user_id}** on thread **{thread_id}** · model: **{st.session_state.model}**")

chat_col, mem_col = st.columns([3, 2], gap="large")

with chat_col:
    tab_chat, tab_tests = st.tabs(["Chat", "Scripted tests"])

    with tab_chat:
        # The chat history shown here is read back from the checkpointer, not from Streamlit.
        history = graph.get_state(config).values.get("messages", [])
        box = st.container(height=460)
        with box:
            if not history:
                st.info("New thread: short-term memory is empty. Long-term memory (right) still applies.")
            for m in history:
                role = "user" if isinstance(m, HumanMessage) else "assistant"
                if isinstance(m, (HumanMessage, AIMessage)):
                    st.chat_message(role).write(m.content)

        st.caption("Try: “Hi, I'm a data analyst.” → New thread → “Explain GROUP BY.” → “Actually, I teach now.” → “Forget my preferences.”")
        if prompt := st.chat_input("Message the tutor"):
            with st.spinner("Thinking..."):
                try:
                    graph.invoke(
                        {"messages": [HumanMessage(prompt)]},
                        config,
                        context=md.Context(user_id=user_id),
                    )
                except Exception as e:
                    st.error(f"Model call failed: {e}")
            st.rerun()

    with tab_tests:
        st.write("Runs the five tests on a fresh graph: same thread, new thread, correction, isolation, forget.")
        if st.button("Run the five tests", type="primary"):
            buf = io.StringIO()
            with st.spinner("Running..."), contextlib.redirect_stdout(buf):
                try:
                    md.run_tests()
                except Exception as e:
                    print(f"\nERROR: {e}")
            st.session_state.test_log = buf.getvalue()
        if log := st.session_state.get("test_log"):
            st.code(log, language="text")

with mem_col:
    st.subheader("Short-term memory")
    st.caption("Checkpointer · keyed by thread_id")
    st.metric("Messages in this thread", len(history))
    st.code(f'thread_id = "{user_id}:{thread_id}"', language="python")

    st.subheader("Long-term memory")
    st.caption("Store · namespaced by user_id")
    st.code(f"namespace = {ns}", language="python")
    item = store.get(ns, md.PROFILE_KEY)
    if item:
        st.json(item.value)
        if st.button("Forget this learner"):
            store.delete(ns, md.PROFILE_KEY)
            st.rerun()
    else:
        st.info("Nothing stored for this learner yet.")

    with st.expander("Everything in the store (all learners)"):
        rows = [
            {"namespace": " / ".join(i.namespace), "key": i.key, "value": json.dumps(i.value)}
            for i in store.search(("learners",))
        ]
        if rows:
            st.dataframe(rows, hide_index=True, width="stretch")
        else:
            st.write("Empty.")
