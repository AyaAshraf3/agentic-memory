# Live demo: agent memory with LangGraph

## Setup (once)

```bash
py -3 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      # then paste your LiteLLM key into .env
```

## Browser UI (Streamlit)

```bash
streamlit run app.py
```

- **Sidebar:** pick the model (listed live from the LiteLLM proxy, or "Offline tutor"), the learner, and "New" for a new thread.
- **Chat tab:** the history is read back from the checkpointer.
- **Scripted tests tab:** runs the five tests below.
- **Right panel:** the thread's message count (short-term) and the learner's profile in the store (long-term), plus a "Forget this learner" button.

## Run

```bash
# Five scripted tests, fully offline (no API key needed)
python memory_demo.py

# Same tests with a real model
set MODEL=anthropic:claude-sonnet-5        # PowerShell: $env:MODEL="anthropic:claude-sonnet-5"
set ANTHROPIC_API_KEY=...
python memory_demo.py

# Free chat: /store shows memory, /thread t2 switches conversation, /user u_omar switches learner
python memory_demo.py --chat --user u_maya --thread t1

# Optional: the same idea with Mem0
pip install mem0ai
python mem0_demo.py
```

## What the five tests show

| Test | What to point at |
|---|---|
| 1. Same conversation | The checkpointer holds 4 messages for thread t1 |
| 2. New conversation | A new thread still knows the role, read from the store |
| 3. Correction | The profile is overwritten, not duplicated |
| 4. Isolation | Omar's namespace is empty; Maya's data never appears |
| 5. Forget | The preferences field disappears from the store |
