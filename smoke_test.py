from rag_engine import TalentSearch
from llm import LLMClient
import bias

ts = TalentSearch()
assert ts.col.count() > 0, "Index is empty, run ingest.py first"

# Retrieval
q = "Find me a Junior Data Analyst who knows SQL and Tableau"
top = ts.search(q, k=3)
assert len(top) == 3
for c in top:
    print(c["id"], c["score"], "matched:", c["matched"], "missing:", c["missing"])

# LLM step (falls back to rule-based text if no key is set)
import os
llm = LLMClient(os.getenv("OPENAI_API_KEY"), None, "gpt-4o-mini")
print(llm.evaluate_candidates(q, top, {c["id"]: c["text"] for c in top})[:600])

# Chat retrieval
print(ts.search_chunks("Does this candidate have leadership experience?", n=2, resume_id=top[0]["id"]))

# Bias tools
a = bias.audit_retrieval(ts, k=10)
print("Audit:", a["note"])
r = bias.counterfactual_test(ts, q, [c["id"] for c in top], names_per_group=2)
print("Name-swap spread:", r["spread"], "flag:", r["flag"])
print("ALL OK")