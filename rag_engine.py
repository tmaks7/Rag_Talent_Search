"""Vector index + hybrid retrieval over resumes (ChromaDB + sentence-transformers)."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import chromadb

import config
from bias import anonymize
from data_loader import Resume

BUILTIN_SKILLS = [
    "SQL", "Tableau", "Power BI", "Excel", "Python", "R Studio", "Java", "JavaScript", "TypeScript",
    "C++", "C#", "Go", "Scala", "React", "Angular", "Node.js", "Django", "Flask", "Spring",
    "Docker", "Kubernetes", "Terraform", "AWS", "Azure", "GCP", "Linux", "Git", "CI/CD", "Jira",
    "Agile", "Scrum", "Pandas", "NumPy", "scikit-learn", "TensorFlow", "PyTorch", "NLP",
    "Machine Learning", "Deep Learning", "Statistics", "Looker", "Spark", "Hadoop", "Airflow",
    "MongoDB", "PostgreSQL", "MySQL", "Oracle", "QuickBooks", "SAP", "Auditing", "IFRS",
    "Recruitment", "Payroll", "Budgeting", "HTML", "CSS", "Figma", "REST APIs", "Leadership",
    "Stakeholder Management", "Tax Compliance", "Onboarding", "HRIS",
]
_STOP = {"and", "the", "with", "in", "of", "to", "for", "a", "an", "knows", "know", "who", "find",
         "me", "experience", "skills", "years", "year", "less", "than", "junior", "senior"}


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    text = re.sub(r"[ \t]+", " ", text).strip()
    if len(text) <= size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            brk = max(text.rfind("\n", start, end), text.rfind(". ", start, end))
            if brk > start + size * 0.5:
                end = brk + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


def skill_in(skill: str, text: str) -> bool:
    return re.search(rf"(?<![\w+#]){re.escape(skill)}(?![\w+#])", text, re.I) is not None


class TalentSearch:
    def __init__(self, persist_dir: str = config.CHROMA_DIR, embed_model: str = config.EMBED_MODEL):
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self.store_path = Path(persist_dir) / "resumes.json"
        self.embed_model = embed_model
        self._model = None
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.col = self.client.get_or_create_collection(
            config.COLLECTION, metadata={"hnsw:space": "cosine"})
        self.resumes: dict = json.loads(self.store_path.read_text()) if self.store_path.exists() else {}
        self.skill_vocab = self._build_vocab()

    # ---------------------------------------------------------------- embeddings
    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.embed_model)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()

    # ---------------------------------------------------------------- indexing
    def index(self, resumes: list[Resume], anonymize_embeddings: bool = False) -> dict:
        try:
            self.client.delete_collection(config.COLLECTION)
        except Exception:
            pass
        self.col = self.client.create_collection(config.COLLECTION, metadata={"hnsw:space": "cosine"})
        ids, docs, metas = [], [], []
        for r in resumes:
            text = anonymize(r.text, r.name) if anonymize_embeddings else r.text
            for i, ch in enumerate(chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)):
                ids.append(f"{r.id}::{i}")
                docs.append(ch)
                metas.append({"resume_id": r.id, "chunk": i})
        for b in range(0, len(docs), 128):
            sl = slice(b, b + 128)
            self.col.add(ids=ids[sl], documents=docs[sl], metadatas=metas[sl],
                         embeddings=self.embed(docs[sl]))
        self.resumes = {r.id: {"text": r.text, "entities": r.entities} for r in resumes}
        self.store_path.write_text(json.dumps(self.resumes))
        self.skill_vocab = self._build_vocab()
        return {"resumes": len(resumes), "chunks": len(docs)}

    def add_resumes(self, resumes: list[Resume], anonymize_embeddings: bool = False) -> dict:
        """Add (or replace, by id) resumes WITHOUT touching the rest of the index."""
        ids, docs, metas = [], [], []
        for r in resumes:
            try:
                self.col.delete(where={"resume_id": r.id})  # re-upload of the same file replaces it
            except Exception:
                pass
            text = anonymize(r.text, r.name) if anonymize_embeddings else r.text
            for i, ch in enumerate(chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)):
                ids.append(f"{r.id}::{i}")
                docs.append(ch)
                metas.append({"resume_id": r.id, "chunk": i})
        for b in range(0, len(docs), 128):
            sl = slice(b, b + 128)
            self.col.add(ids=ids[sl], documents=docs[sl], metadatas=metas[sl],
                         embeddings=self.embed(docs[sl]))
        for r in resumes:
            self.resumes[r.id] = {"text": r.text, "entities": r.entities}
        self.store_path.write_text(json.dumps(self.resumes))
        self.skill_vocab = self._build_vocab()
        return {"resumes": len(resumes), "chunks": len(docs)}

    # ---------------------------------------------------------------- skills
    def _build_vocab(self) -> dict[str, str]:
        vocab = {s.lower(): s for s in BUILTIN_SKILLS}
        freq: Counter = Counter()
        for e in self.resumes.values():
            seen = set()
            for blob in e["entities"].get("Skills", []):
                for part in re.split(r"[\n,;•●·|]", blob):
                    s = re.sub(r"\(.*?\)", "", part).strip(" -:.")
                    if 2 <= len(s) <= 25 and re.match(r"^[A-Za-z][A-Za-z0-9+#./ -]*$", s) \
                            and s.lower() not in _STOP:
                        seen.add(s)
            freq.update(seen)
        for s, n in freq.items():
            if n >= 3:
                vocab.setdefault(s.lower(), s)
        return vocab

    def extract_skills(self, query: str) -> list[str]:
        found = [disp for low, disp in self.skill_vocab.items() if skill_in(low, query)]
        # drop terms contained inside a longer matched term (e.g. "SQL" inside "PostgreSQL" is
        # already blocked by the word-boundary rule; this handles "Power" vs "Power BI")
        return [s for s in found if not any(s != o and s.lower() in o.lower() for o in found)]

    # ---------------------------------------------------------------- retrieval
    def display_name(self, rid: str) -> str:
        ents = self.resumes.get(rid, {}).get("entities", {})
        n = ents.get("Name") or ents.get("Source file")
        return n[0] if n else rid

    def search(self, query: str, k: int = config.TOP_K, pool: int = config.CANDIDATE_POOL) -> list[dict]:
        """Semantic chunk search, rolled up to resumes, re-ranked by required-skill coverage."""
        total = self.col.count()
        if total == 0:
            return []
        res = self.col.query(query_embeddings=self.embed([query]), n_results=min(total, pool * 4),
                             include=["documents", "metadatas", "distances"])
        by_resume: dict[str, list] = {}
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            by_resume.setdefault(meta["resume_id"], []).append((1 - dist, doc))
        required = self.extract_skills(query)
        out = []
        for rid, hits in by_resume.items():
            hits.sort(key=lambda h: h[0], reverse=True)
            semantic = hits[0][0] if len(hits) == 1 else 0.7 * hits[0][0] + 0.3 * hits[1][0]
            text = self.resumes[rid]["text"]
            matched = [s for s in required if skill_in(s, text)]
            coverage = len(matched) / len(required) if required else None
            final = semantic if coverage is None else 0.5 * semantic + 0.5 * coverage
            out.append({"id": rid, "name": self.display_name(rid), "score": round(final, 4),
                        "semantic": round(semantic, 4), "coverage": coverage,
                        "required": required, "matched": matched,
                        "missing": [s for s in required if s not in matched],
                        "text": text, "entities": self.resumes[rid]["entities"],
                        "snippets": [h[1] for h in hits[:2]]})
        out.sort(key=lambda c: c["score"], reverse=True)
        return out[:k]

    def search_chunks(self, question: str, n: int = 5, resume_id: str | None = None) -> list[dict]:
        if self.col.count() == 0:
            return []
        kwargs = {"where": {"resume_id": resume_id}} if resume_id else {}
        res = self.col.query(query_embeddings=self.embed([question]), n_results=n,
                             include=["documents", "metadatas", "distances"], **kwargs)
        return [{"resume_id": m["resume_id"], "text": d, "score": round(1 - dist, 4)}
                for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]
