"""LLM layer: shortlist evaluation and chat answers. Works with any OpenAI-compatible API
(OpenAI, Gemini's OpenAI endpoint, Ollama/Llama 3, vLLM...). Falls back to rule-based text."""
from __future__ import annotations

import config

EVAL_SYSTEM = (
    "You are an assistant to a technical recruiter. Judge candidates ONLY from the resume text "
    "provided and the recruiter's request. Cite concrete evidence from the resume. If something "
    "the request needs is not evidenced, say 'not evidenced' rather than guessing. Never infer, "
    "mention or weigh gender, age, ethnicity, nationality, religion, marital status or any other "
    "protected trait, and ignore names when judging fit. Be concise.")

CHAT_SYSTEM = (
    "You answer a recruiter's questions about candidates using ONLY the resume excerpts given. "
    "Refer to candidates by their ID. If the excerpts do not contain the answer, say it is not "
    "evidenced. Quote or point to specific evidence. Never infer or discuss protected traits "
    "(gender, age, ethnicity, religion, nationality, marital status).")


class LLMClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 model: str = "gpt-4o-mini"):
        self.model, self.client, self.error = model, None, None
        if api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key, base_url=base_url or None)
            except ImportError:
                self.error = "openai package not installed"

    @property
    def available(self) -> bool:
        return self.client is not None

    def complete(self, messages: list[dict], temperature: float = 0.2, max_tokens: int = 1200) -> str:
        r = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        return (r.choices[0].message.content or "").strip()

    # ------------------------------------------------------------ shortlist evaluation
    def evaluate_candidates(self, query: str, candidates: list[dict], texts: dict[str, str]) -> str:
        """`texts` maps candidate id -> resume text to show the LLM (possibly redacted)."""
        if not self.available:
            return rule_based_summary(query, candidates)
        blocks = "\n\n".join(
            f"### Candidate {i} (ID: {c['id']})\n{texts[c['id']][:config.MAX_RESUME_CHARS]}"
            for i, c in enumerate(candidates, 1))
        user = (f"Recruiter request: {query}\n\nResumes:\n\n{blocks}\n\n"
                "For EACH candidate write:\n"
                "**Candidate N (ID) - Strong / Partial / Weak fit**\n"
                "- Why they fit: 2-3 bullets with evidence from the resume\n"
                "- Gaps: what the request needs that is not evidenced\n\n"
                "Finish with **Recommendation:** one short paragraph comparing the candidates.")
        try:
            return self.complete([{"role": "system", "content": EVAL_SYSTEM},
                                  {"role": "user", "content": user}])
        except Exception as e:  # network, auth, model name...
            return f"> LLM call failed ({type(e).__name__}: {e}). Rule-based summary instead.\n\n" \
                   + rule_based_summary(query, candidates)

    # ------------------------------------------------------------ chat
    def answer(self, question: str, chunks: list[dict], history: list[dict]) -> str:
        if not self.available:
            return rule_based_answer(chunks)
        ctx = "\n\n".join(f"[Candidate {c['resume_id']}]\n{c['text']}" for c in chunks) or "(none)"
        msgs = [{"role": "system", "content": CHAT_SYSTEM}] + history[-6:] + [
            {"role": "user", "content": f"Resume excerpts:\n{ctx}\n\nQuestion: {question}"}]
        try:
            return self.complete(msgs)
        except Exception as e:
            return f"> LLM call failed ({type(e).__name__}: {e}).\n\n" + rule_based_answer(chunks)


# ---------------------------------------------------------------- fallbacks
def rule_based_summary(query: str, candidates: list[dict]) -> str:
    lines = ["> No LLM configured: showing a rule-based summary. Add an API key (or run Llama 3 "
             "via Ollama) in the sidebar for written fit explanations.\n"]
    for i, c in enumerate(candidates, 1):
        e = c["entities"]
        role = (e.get("Designation") or ["role not listed"])[0]
        yrs = (e.get("Years of Experience") or ["n/a"])[0]
        verdict = ("Strong" if c["coverage"] == 1 else "Partial" if c["coverage"] else "Weak") \
            if c["coverage"] is not None else "Semantic match"
        lines.append(f"**Candidate {i} ({c['id']}) - {verdict}**")
        lines.append(f"- Role: {role} | Experience: {yrs}")
        if c["required"]:
            lines.append(f"- Matched skills: {', '.join(c['matched']) or 'none'}")
            lines.append(f"- Not evidenced: {', '.join(c['missing']) or 'none'}")
        lines.append(f"- Relevance score: {c['score']}\n")
    return "\n".join(lines)


def rule_based_answer(chunks: list[dict]) -> str:
    if not chunks:
        return "No relevant resume text found."
    out = ["> No LLM configured. Most relevant resume excerpts:\n"]
    out += [f"**{c['resume_id']}**: {c['text'][:400]}..." for c in chunks[:3]]
    return "\n\n".join(out)
