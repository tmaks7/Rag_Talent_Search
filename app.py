"""Streamlit UI.   streamlit run app.py"""
import os

import pandas as pd
import streamlit as st

import bias
import config
from data_loader import load_resumes, resume_from_file
from llm import LLMClient
from rag_engine import TalentSearch

st.set_page_config(page_title="Talent Search RAG", page_icon="🔎", layout="wide")

PROVIDERS = {
    "OpenAI": {"base_url": None, "model": "gpt-4o-mini", "env": "OPENAI_API_KEY"},
    "Gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
               "model": "gemini-2.0-flash", "env": "GEMINI_API_KEY"},
    "Ollama (Llama 3, local)": {"base_url": "http://localhost:11434/v1", "model": "llama3",
                                "env": None, "key": "ollama"},
    "None (rule-based fallback)": None,
}


@st.cache_resource(show_spinner="Loading embedding model and vector store...")
def get_engine() -> TalentSearch:
    return TalentSearch()


@st.cache_resource
def get_llm(api_key, base_url, model) -> LLMClient:
    return LLMClient(api_key, base_url, model)


engine = get_engine()
ss = st.session_state
ss.setdefault("candidates", [])
ss.setdefault("summary", "")
ss.setdefault("query", "")
ss.setdefault("chats", {})

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("⚙️ Setup")
    st.subheader("1. LLM")
    provider = st.selectbox("Provider", list(PROVIDERS))
    cfg = PROVIDERS[provider]
    if cfg:
        default_key = cfg.get("key") or os.getenv(cfg["env"] or "", "")
        api_key = st.text_input("API key", value=default_key, type="password")
        model = st.text_input("Model", value=cfg["model"])
        base_url = st.text_input("Base URL", value=cfg["base_url"] or "")
        llm = get_llm(api_key or None, base_url or None, model)
    else:
        llm = get_llm(None, None, "")
    st.caption("LLM ready ✅" if llm.available else "No LLM: rule-based summaries")

    st.subheader("2. Index")
    st.caption(f"Indexed chunks: **{engine.col.count()}** · resumes: **{len(engine.resumes)}**")
    upload = st.file_uploader("Resume dataset (.json / .csv)", type=["json", "csv"])
    data_path = config.DATA_PATH
    if upload:
        config.DATA_DIR.mkdir(exist_ok=True)
        data_path = str(config.DATA_DIR / upload.name)
        with open(data_path, "wb") as f:
            f.write(upload.getbuffer())
    anon_embed = st.checkbox("Anonymise before embedding", value=False,
                             help="Redacts names, contacts and gendered pronouns before the "
                                  "text is embedded, so identity cannot influence retrieval.")
    if st.button("Build / rebuild index", type="primary"):
        with st.spinner("Embedding resumes..."):
            resumes, source = load_resumes(data_path)
            stats = engine.index(resumes, anonymize_embeddings=anon_embed)
        ss.candidates, ss.summary, ss.chats = [], "", {}
        st.success(f"Indexed {source}: {stats['chunks']} chunks.")
        if "SYNTHETIC" in source:
            st.warning("Using synthetic demo data. Put the Kaggle file in data/ or upload it.")

    st.markdown("**Add applicant resumes** (keeps the existing index)")
    new_files = st.file_uploader("PDF, DOCX or TXT", type=["pdf", "docx", "txt"],
                                 accept_multiple_files=True, key="resume_files")
    if new_files and st.button("Add to index"):
        added, problems = [], []
        for f in new_files:
            try:
                added.append(resume_from_file(f.name, f.getvalue()))
            except Exception as e:
                problems.append(f"{f.name}: {e}")
        if added:
            with st.spinner("Embedding..."):
                stats = engine.add_resumes(added, anonymize_embeddings=anon_embed)
            ss.chats = {}
            st.success(f"Added {stats['resumes']} resume(s) ({stats['chunks']} chunks). "
                       f"Index now holds {len(engine.resumes)} resumes.")
            st.caption("IDs: " + ", ".join(r.id for r in added))
        for p in problems:
            st.error(p)

    st.subheader("3. Fairness")
    blind = st.checkbox("Blind screening for the LLM", value=True,
                        help="Redacts name, contacts and gendered pronouns from resume text "
                             "before it is sent to the LLM or shown in the UI.")

st.title("RAG Talent Search")
if engine.col.count() == 0:
    st.info("Build the index from the sidebar first (a synthetic demo set is used if no dataset is found).")
    st.stop()


def view(c: dict) -> str:
    n = (c["entities"].get("Name") or [None])[0]
    return bias.anonymize(c["text"], n) if blind else c["text"]


tab_search, tab_chat, tab_bias = st.tabs(["🔍 Search", "💬 Chat with resumes", "⚖️ Bias check"])

# ------------------------------------------------------------------ search
with tab_search:
    with st.form("search"):
        q = st.text_input("Describe who you need",
                          placeholder="Find me a Junior Data Analyst who knows SQL and Tableau")
        go = st.form_submit_button("Search", type="primary")
    if go and q.strip():
        with st.spinner("Retrieving candidates and asking the LLM to assess the top 3..."):
            cands = engine.search(q, k=config.TOP_K)
            texts = {c["id"]: view(c) for c in cands}
            ss.summary = llm.evaluate_candidates(q, cands, texts)
            ss.candidates, ss.query = cands, q
    if ss.candidates:
        st.subheader(f"Why these {len(ss.candidates)} fit")
        st.markdown(ss.summary)
        st.subheader("Shortlist details")
        for c in ss.candidates:
            label = c["id"] if blind else f"{c['name']} ({c['id']})"
            with st.expander(f"{label} · score {c['score']}"):
                m1, m2, m3 = st.columns(3)
                m1.metric("Final score", c["score"])
                m2.metric("Semantic similarity", c["semantic"])
                m3.metric("Skill coverage", "n/a" if c["coverage"] is None else f"{c['coverage']:.0%}")
                if c["required"]:
                    st.write("✅ Matched:", ", ".join(c["matched"]) or "none")
                    st.write("❌ Not found:", ", ".join(c["missing"]) or "none")
                st.markdown("**Best-matching passage**")
                st.text(bias.anonymize(c["snippets"][0], (c["entities"].get("Name") or [None])[0])
                        if blind else c["snippets"][0])
                with st.popover("Full resume"):
                    st.text(view(c))

# ------------------------------------------------------------------ chat
with tab_chat:
    st.caption("Ask follow-ups, e.g. “Does this candidate have leadership experience?”")
    options = {"All indexed resumes": None}
    for c in ss.candidates:
        options[f"{c['id']} (shortlist #{ss.candidates.index(c) + 1})"] = c["id"]
    scope = st.selectbox("Ask about", list(options))
    rid = options[scope]
    history = ss.chats.setdefault(scope, [])
    for m in history:
        st.chat_message(m["role"]).markdown(m["content"])
    question = st.chat_input("Ask about the candidates...")
    if question:
        st.chat_message("user").markdown(question)
        chunks = engine.search_chunks(question, n=5 if rid else 6, resume_id=rid)
        if blind:
            for ch in chunks:
                n = (engine.resumes[ch["resume_id"]]["entities"].get("Name") or [None])[0]
                ch["text"] = bias.anonymize(ch["text"], n)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                ans = llm.answer(question, chunks, history)
            st.markdown(ans)
            with st.expander("Evidence retrieved"):
                for ch in chunks:
                    st.markdown(f"**{ch['resume_id']}** (similarity {ch['score']})")
                    st.text(ch["text"])
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": ans}]

# ------------------------------------------------------------------ bias
with tab_bias:
    st.markdown(
        "Two audits using **proxies** (pronouns and first names), since the dataset has no "
        "demographic labels. A flag means *investigate*, not *proven bias*.")
    st.subheader("1. Retrieval adverse-impact check")
    k_bias = st.slider("Top-k per query", 5, 20, 10)
    if st.button("Run retrieval audit"):
        with st.spinner("Running audit queries..."):
            a = bias.audit_retrieval(engine, k=k_bias)
        (st.error if a["flag"] else st.success)(a["note"])
        st.write("Inferred gender counts in database:", a["pool"])
        st.write("Selection rates:", {g: round(r, 4) for g, r in a["rates"].items()})
        st.dataframe(pd.DataFrame(a["rows"]), use_container_width=True)

    st.subheader("2. Name-swap counterfactual test")
    st.caption("Same resume text, only the name changes. Similarity shifts mean the retriever reacts to names.")
    cf_query = st.text_input("Query to test", value=ss.query or "Junior Data Analyst with SQL and Tableau")
    thr = st.slider("Flag if score spread exceeds", 0.002, 0.05, 0.01, 0.002)
    if st.button("Run name-swap test"):
        ids = [c["id"] for c in engine.search(cf_query, k=10)]
        with st.spinner(f"Embedding {len(ids) * 36} name variants..."):
            r = bias.counterfactual_test(engine, cf_query, ids, spread_threshold=thr)
        msg = f"Spread between best and worst group: {r['spread']} (threshold {thr})."
        (st.error if r["flag"] else st.success)(
            msg + (" Names are moving the score." if r["flag"] else " No meaningful name effect."))
        st.dataframe(pd.DataFrame(r["table"]).sort_values("mean similarity", ascending=False),
                     use_container_width=True)
