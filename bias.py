"""Bias tooling: identity redaction, a demographic *proxy* inference, and two audits.

IMPORTANT: demographics are never stored in the dataset. Everything here uses crude
proxies (pronouns, first names). Treat results as a prompt to investigate, not proof.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

# ---------------------------------------------------------------- name lists
# Name-coded groups, as used in classic résumé audit studies. Proxies only.
NAMES = {
    "Anglo": {
        "F": ["Emily", "Sarah", "Jessica", "Laura", "Megan"],
        "M": ["James", "Michael", "Daniel", "Andrew", "Matthew"],
        "L": ["Smith", "Johnson", "Williams", "Brown", "Taylor"],
    },
    "Nigerian": {
        "F": ["Chioma", "Adaeze", "Funke", "Ngozi", "Aisha"],
        "M": ["Chinedu", "Emeka", "Tunde", "Ibrahim", "Obinna"],
        "L": ["Okafor", "Adeyemi", "Balogun", "Eze", "Nwosu"],
    },
    "Indian": {
        "F": ["Priya", "Ananya", "Kavita", "Sneha", "Meera"],
        "M": ["Rahul", "Amit", "Arjun", "Vikram", "Sanjay"],
        "L": ["Sharma", "Patel", "Iyer", "Reddy", "Gupta"],
    },
    "East/SE Asian": {
        "F": ["Mei", "Yuki", "Jiwoo", "Lina", "Hana"],
        "M": ["Wei", "Kenji", "Minjun", "Hiroshi", "Jun"],
        "L": ["Chen", "Tanaka", "Kim", "Wang", "Nguyen"],
    },
    "Arabic": {
        "F": ["Fatima", "Layla", "Noor", "Amira", "Salma"],
        "M": ["Omar", "Hassan", "Khalid", "Yusuf", "Karim"],
        "L": ["Hussein", "Mansour", "Nasser", "Farouk", "Haddad"],
    },
    "Hispanic": {
        "F": ["Maria", "Sofia", "Camila", "Lucia", "Valentina"],
        "M": ["Carlos", "Mateo", "Diego", "Javier", "Luis"],
        "L": ["Garcia", "Martinez", "Lopez", "Hernandez", "Rodriguez"],
    },
}
FEMALE_FIRST = {n.lower() for g in NAMES.values() for n in g["F"]} | {
    "anna", "mary", "linda", "susan", "grace", "blessing", "joy", "ruth", "esther", "sara"}
MALE_FIRST = {n.lower() for g in NAMES.values() for n in g["M"]} | {
    "john", "david", "peter", "paul", "mark", "samuel", "joseph", "david", "ahmed", "mohammed"}

_PRON_F = re.compile(r"\b(she|her|hers|herself)\b", re.I)
_PRON_M = re.compile(r"\b(he|him|his|himself)\b", re.I)


def infer_gender(text: str, name: str | None = None) -> str:
    """Pronoun count first, then first-name lookup. Returns female / male / unknown."""
    f, m = len(_PRON_F.findall(text)), len(_PRON_M.findall(text))
    if f != m:
        return "female" if f > m else "male"
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    token = ((name or first_line).split() or [""])[0].lower().strip(".,")
    if token in FEMALE_FIRST:
        return "female"
    if token in MALE_FIRST:
        return "male"
    return "unknown"


# ---------------------------------------------------------------- anonymiser
_PRON_MAP = {"he": "they", "she": "they", "him": "them", "her": "their", "his": "their",
             "hers": "theirs", "himself": "themself", "herself": "themself"}
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"\+?\d[\d\s().-]{7,}\d")
_TITLES = re.compile(r"\b(Mr|Mrs|Ms|Miss|Mx)\.?\s+", re.I)
_PERSONAL_LINES = re.compile(
    r"(?im)^\s*(gender|sex|marital status|date of birth|dob|age|nationality|religion)\s*:.*$")


def _pron(m: re.Match) -> str:
    w = m.group(0)
    r = _PRON_MAP[w.lower()]
    return r.capitalize() if w[0].isupper() else r


_VERB = {"is": "are", "was": "were", "has": "have"}


def _they_verb(m: re.Match) -> str:
    subj = "They" if m.group(1)[0].isupper() else "they"
    return f"{subj} {_VERB[m.group(2).lower()]}"


def _phone(m: re.Match) -> str:
    return "[PHONE]" if sum(c.isdigit() for c in m.group(0)) >= 9 else m.group(0)


def anonymize(text: str, name: str | None = None) -> str:
    """Blind-screening redaction: name, contact details, titles, gendered pronouns."""
    out = _EMAIL.sub("[EMAIL]", text)  # first, so the name pass cannot mangle addresses
    if name:
        parts = [name.strip()] + [p for p in re.split(r"\s+", name.strip()) if len(p) > 1]
        for p in parts:
            out = re.sub(rf"\b{re.escape(p)}\b", "[CANDIDATE]", out, flags=re.I)
    out = _PERSONAL_LINES.sub("", out)
    out = _PHONE.sub(_phone, out)
    out = _TITLES.sub("", out)
    # verb agreement for the singular they: "He is" -> "They are", "she has" -> "they have"
    out = re.sub(r"\b(he|she) (is|was|has)\b", _they_verb, out, flags=re.I)
    out = re.sub(r"\b(she|her|hers|herself|he|him|his|himself)\b", _pron, out, flags=re.I)
    return out


def _name_of(entry: dict) -> str | None:
    n = entry["entities"].get("Name")
    return n[0] if n else None


# ---------------------------------------------------------------- audit 1
DEFAULT_AUDIT_QUERIES = [
    "Junior Data Analyst with SQL and Tableau",
    "Software engineer with Java and cloud experience",
    "Project manager with leadership experience and Agile",
    "Data scientist with Python and machine learning",
    "Accountant with auditing and Excel skills",
    "HR manager experienced in recruitment",
    "Frontend developer with React and JavaScript",
    "DevOps engineer with Kubernetes and Terraform",
]


def audit_retrieval(engine, queries=None, k: int = 10, threshold: float = 0.8) -> dict:
    """Adverse-impact check on retrieval.

    For every query, take the top-k candidates, count inferred gender, then compare each
    group's selection rate (selected / group size in the database) across all queries.
    Ratio of lowest to highest rate below `threshold` (the "four-fifths rule") raises a flag.
    """
    queries = queries or DEFAULT_AUDIT_QUERIES
    genders = {rid: infer_gender(e["text"], _name_of(e)) for rid, e in engine.resumes.items()}
    pool = Counter(g for g in genders.values() if g != "unknown")
    selected, rows = Counter(), []
    for q in queries:
        top = engine.search(q, k=k)
        gs = [genders[c["id"]] for c in top]
        cnt = Counter(gs)
        selected.update({g: n for g, n in cnt.items() if g != "unknown"})
        known = cnt["female"] + cnt["male"]
        rows.append({"query": q, "female": cnt["female"], "male": cnt["male"],
                     "unknown": cnt["unknown"],
                     "female share of known": round(cnt["female"] / known, 2) if known else None})
    rates = {g: selected[g] / (pool[g] * len(queries)) for g in pool if pool[g] > 0}
    result = {"rows": rows, "pool": dict(pool), "rates": rates, "ratio": None,
              "flag": False, "favored": None, "note": ""}
    if len(rates) < 2 or max(rates.values()) == 0:
        result["note"] = "Not enough resumes with an inferable gender to compare groups."
        return result
    hi, lo = max(rates, key=rates.get), min(rates, key=rates.get)
    ratio = rates[lo] / rates[hi]
    result.update(ratio=round(ratio, 3), favored=hi, flag=ratio < threshold)
    result["note"] = (
        f"Selection-rate ratio {ratio:.2f} ({lo} vs {hi}). "
        + ("Below the 0.8 rule of thumb: investigate. The skew may also reflect genuine role "
           "mix in the data, not only model preference." if ratio < threshold
           else "Within the 0.8 rule of thumb."))
    return result


# ---------------------------------------------------------------- audit 2
def counterfactual_test(engine, query: str, resume_ids: list[str], names_per_group: int = 3,
                        spread_threshold: float = 0.01, chunk_chars: int = 900) -> dict:
    """Name-swap test. Identical resume text, only the name changes (pronouns neutralised).

    If the embedding similarity to the query moves with the name's gender/origin, the
    retriever is reacting to identity rather than qualifications.
    """
    qv = np.array(engine.embed([query])[0])
    texts, labels = [], []
    for rid in resume_ids:
        entry = engine.resumes[rid]
        head = anonymize(entry["text"], _name_of(entry))[:chunk_chars]
        for origin, d in NAMES.items():
            for g in ("F", "M"):
                for i in range(names_per_group):
                    full = f"{d[g][i]} {d['L'][i]}"
                    texts.append(f"{full}\n" + head.replace("[CANDIDATE]", full))
                    labels.append((origin, g))
    scores = np.array(engine.embed(texts)) @ qv
    by_group: dict = {}
    for (origin, g), s in zip(labels, scores):
        by_group.setdefault((origin, g), []).append(float(s))
    overall = float(np.mean(scores))
    table = [{"name origin": o, "gender-coded": "female" if g == "F" else "male",
              "mean similarity": round(float(np.mean(v)), 4),
              "delta vs overall": round(float(np.mean(v)) - overall, 4)}
             for (o, g), v in by_group.items()]
    means = [t["mean similarity"] for t in table]
    spread = round(max(means) - min(means), 4)
    return {"table": table, "spread": spread, "threshold": spread_threshold,
            "flag": spread > spread_threshold, "n_resumes": len(resume_ids)}
