"""
Optional add-on: the same idea with Mem0 as a plug-in memory layer.

pip install mem0ai
set OPENAI_API_KEY=...        (Mem0's default extractor and embeddings use OpenAI)
python mem0_demo.py
"""

from mem0 import Memory

memory = Memory()

memory.add("Hi, I'm Maya. I'm a data analyst and I prefer short answers.", user_id="u_maya")
memory.add("I'm Omar, I teach high-school maths.", user_id="u_omar")

for user in ("u_maya", "u_omar"):
    hits = memory.search("What is this learner's job?", user_id=user)
    results = hits["results"] if isinstance(hits, dict) else hits
    print(user, "->", [h["memory"] for h in results])

print("All of Maya's memories:", memory.get_all(user_id="u_maya"))
