# RAG-Powered Talent Search Engine

Ask for candidates in plain English ("Find me a Junior Data Analyst who knows SQL and Tableau"),
get the top 3 resumes with an LLM-written explanation of why they fit, then chat with the
resume database. Includes a bias audit.

## Quick start

```bash
pip install -r requirements.txt

# 1. Get the data (optional: a synthetic demo set is used if absent)
#    Kaggle -> "Resume Entities for NER" -> save as data/Entity Recognition in Resumes.json

# 2. Choose an LLM (any OpenAI-compatible endpoint)
export OPENAI_API_KEY=sk-...        # or GEMINI_API_KEY, or run `ollama run llama3`

# 3. Launch
streamlit run app.py                # click "Build / rebuild index" in the sidebar
# or build from the CLI first:  python ingest.py --data "data/Entity Recognition in Resumes.json"
```

No API key? The app still runs and falls back to rule-based summaries.

## Architecture

```
Kaggle JSON ─> data_loader ─> chunk (900 chars, 150 overlap) ─> MiniLM embeddings ─> ChromaDB (cosine)
                                                                                       │
query ─> embed ─> top chunks ─> roll up per resume ─> re-rank with skill coverage ─────┘
                                                      │
                          top 3 resumes ─> LLM ("why they fit", evidence, gaps) ─> Streamlit
                          chat question ─> chunks (one candidate or whole DB) ─> LLM answer + evidence
```

| File | Role |
|---|---|
| `data_loader.py` | Parses the NER JSON (keeps entity labels such as Skills, Designation), CSV, or .txt folder; synthetic fallback |
| `rag_engine.py` | Chunking, ChromaDB index, hybrid retrieval: semantic score blended 50/50 with coverage of skills named in the query |
| `llm.py` | OpenAI-compatible client (OpenAI, Gemini, Ollama/Llama 3), evaluation and chat prompts with anti-bias instructions |
| `bias.py` | Anonymiser, gender proxy, retrieval adverse-impact audit, name-swap counterfactual test |
| `app.py` | Streamlit: Search, Chat, Bias check tabs |

### Why hybrid ranking
Pure embeddings blur "knows SQL" with "worked near data". Extracting required skills from the
query and checking them in the resume text makes "SQL and Tableau" behave the way recruiters expect,
while embeddings still handle phrasing like "junior" or "entry-level reporting".

## Bias check (bonus)

The dataset has no demographic labels, so everything uses proxies (pronouns, first names).

1. **Blind screening** — redacts name, email, phone, titles, personal-detail lines and gendered
   pronouns. Applies to LLM prompts and the UI; optionally also before embedding
   ("Anonymise before embedding", needs a rebuild).
2. **Retrieval adverse-impact audit** — runs 8 role queries, counts inferred gender in the top-k,
   and compares selection rates per group. A ratio below 0.8 (four-fifths rule) is flagged.
3. **Name-swap counterfactual** — identical resume text with names from 6 name-origin groups ×
   2 gender codings; reports how much the query similarity moves. A spread above the threshold is flagged.

Limitations: proxies misclassify people; a skewed database (e.g. more women in HR roles) can
look like model bias; the 0.8 rule is a heuristic, not a legal test. Use flags as a reason to investigate.
The LLM stage is guarded by prompt instructions and blind screening but is not separately audited;
extending the name-swap test to LLM verdicts is a natural next step.

## Example queries
- `Find me a Junior Data Analyst who knows SQL and Tableau`
- Chat: `Does this candidate have leadership experience?` · `Which of these has cloud experience?`
