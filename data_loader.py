"""Load resumes from the Kaggle 'Resume Entities for NER' file (or CSV / folder of .txt),
with a synthetic fallback so the app runs before you download anything."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from bias import NAMES


@dataclass
class Resume:
    id: str
    text: str
    entities: dict = field(default_factory=dict)

    def first(self, label: str):
        v = self.entities.get(label)
        return v[0] if v else None

    @property
    def name(self):
        return self.first("Name")


# ------------------------------------------------------------------ Kaggle NER
def _parse_ner_record(rec: dict, idx: int) -> Resume:
    ents: dict = {}
    for ann in rec.get("annotation") or []:
        labels = ann.get("label") or []
        label = labels[0] if labels else "UNKNOWN"
        for p in ann.get("points") or []:
            t = (p.get("text") or "").strip()
            if t and t not in ents.setdefault(label, []):
                ents[label].append(t)
    return Resume(id=f"res_{idx:04d}", text=(rec.get("content") or "").strip(), entities=ents)


def load_kaggle_json(path: Path) -> list[Resume]:
    raw = path.read_text(encoding="utf-8", errors="ignore").strip()
    if raw.startswith("["):
        records = json.loads(raw)
    else:  # JSON Lines (the Dataturks format)
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    return [_parse_ner_record(r, i) for i, r in enumerate(records) if r.get("content")]


def load_csv(path: Path) -> list[Resume]:
    out = []
    with path.open(encoding="utf-8", errors="ignore", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            text = row.get("Resume_str") or row.get("text") or row.get("Resume") or ""
            if text.strip():
                ents = {"Designation": [row["Category"]]} if row.get("Category") else {}
                out.append(Resume(id=f"res_{i:04d}", text=text.strip(), entities=ents))
    return out


SUPPORTED_EXT = {".pdf", ".docx", ".txt"}


def extract_text(filename: str, data: bytes) -> str:
    """Plain text from a PDF, DOCX or TXT resume. Raises ValueError with a readable message."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise ValueError("PDF is password-protected")
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
    elif ext == ".docx":
        from docx import Document
        doc = Document(io.BytesIO(data))
        parts = [p.text for p in doc.paragraphs]
        for table in doc.tables:  # many resume templates put skills/experience in tables
            for row in table.rows:
                parts.append(" | ".join(c.text.strip() for c in row.cells))
        text = "\n".join(parts)
    elif ext == ".doc":
        raise ValueError("old .doc format is not supported: re-save it as .docx or PDF")
    elif ext == ".txt":
        text = data.decode("utf-8", errors="ignore")
    else:
        raise ValueError(f"unsupported file type {ext or '(none)'}")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) < 50:
        raise ValueError("no readable text found (a scanned/image-only PDF needs OCR first)")
    return text


def _guess_name(text: str) -> str | None:
    """Treat the first line as the candidate's name if it looks like one (2-4 words, letters only)."""
    first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    if (2 <= len(first.split()) <= 4 and len(first) <= 40
            and re.fullmatch(r"[A-Za-z][A-Za-z .'-]+", first)
            and not re.search(r"resume|curriculum|vitae|profile|summary", first, re.I)):
        return first
    return None


def resume_from_file(filename: str, data: bytes) -> Resume:
    text = extract_text(filename, data)
    ents = {"Source file": [filename]}
    name = _guess_name(text)
    if name:
        ents["Name"] = [name]
    # id comes from the content, so re-uploading the same file replaces it instead of duplicating
    return Resume(id="up_" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:8],
                  text=text, entities=ents)


def load_folder(path: Path) -> list[Resume]:
    out = []
    for p in sorted(path.iterdir()):
        if p.suffix.lower() in SUPPORTED_EXT:
            try:
                out.append(resume_from_file(p.name, p.read_bytes()))
            except Exception as e:
                print(f"Skipped {p.name}: {e}")
    return out


# ------------------------------------------------------------------ synthetic sample
ROLES = {
    "Junior Data Analyst": ["SQL", "Tableau", "Excel", "Python", "Power BI", "Statistics"],
    "Data Analyst": ["SQL", "Tableau", "Excel", "Python", "Power BI", "Looker"],
    "Data Scientist": ["Python", "scikit-learn", "TensorFlow", "NLP", "SQL", "Pandas"],
    "Software Engineer": ["Java", "Spring", "Docker", "AWS", "Git", "REST APIs"],
    "Frontend Developer": ["JavaScript", "React", "TypeScript", "HTML", "CSS", "Figma"],
    "HR Manager": ["Recruitment", "Employee Relations", "Payroll", "Onboarding", "HRIS"],
    "Project Manager": ["Agile", "Scrum", "Jira", "Stakeholder Management", "Budgeting"],
    "Accountant": ["QuickBooks", "Excel", "Auditing", "IFRS", "Tax Compliance"],
    "DevOps Engineer": ["Kubernetes", "Docker", "Terraform", "AWS", "CI/CD", "Linux"],
}
COMPANIES = ["Northwind Labs", "Bluepeak Systems", "Greenfield Analytics", "Atlas Consulting",
             "Meridian Bank", "Orbit Logistics", "Harbor Health", "Sunrise Retail", "Vertex Cloud"]
CITIES = ["Port Harcourt", "Lagos", "London", "Bangalore", "Austin", "Toronto", "Berlin", "Dubai"]
DEGREES = ["B.Sc. Statistics", "B.Sc. Computer Science", "B.A. Economics", "M.Sc. Data Science",
           "B.Eng. Software Engineering", "B.Sc. Accounting", "MBA"]
COLLEGES = ["University of Lagos", "University of Port Harcourt", "IIT Madras",
            "University of Toronto", "TU Berlin", "University of Leeds", "UT Austin"]
BULLETS = ["Built dashboards in {s} to track KPIs for the operations team.",
           "Automated weekly reporting with {s}, saving about 6 hours per week.",
           "Used {s} to analyse customer behaviour and present findings to stakeholders.",
           "Maintained and improved internal tooling based on {s}.",
           "Partnered with product managers to deliver features using {s}."]


def make_sample_resumes(n: int = 36, seed: int = 7) -> list[Resume]:
    rng = random.Random(seed)
    resumes = []
    for i in range(n):
        origin = rng.choice(list(NAMES))
        g = rng.choice("FM")
        d = NAMES[origin]
        first, last = rng.choice(d[g]), rng.choice(d["L"])
        full = f"{first} {last}"
        role = rng.choice(list(ROLES))
        years = rng.randint(0, 2) if role.startswith("Junior") else rng.randint(2, 12)
        pool = ROLES[role]
        skills = rng.sample(pool, k=rng.randint(min(4, len(pool)), len(pool)))
        if role.startswith("Junior") and rng.random() < 0.6:
            skills = list({*skills, "SQL", "Tableau"})
        lead = years >= 6 or rng.random() < 0.3
        subj, poss = ("She", "her") if g == "F" else ("He", "his")
        company, city = rng.choice(COMPANIES), rng.choice(CITIES)
        degree, college = rng.choice(DEGREES), rng.choice(COLLEGES)
        email = f"{first}.{last}@example.com".lower()
        bullets = [rng.choice(BULLETS).format(s=rng.choice(skills)) for _ in range(2)]
        if lead:
            bullets.append(f"Led a team of {rng.randint(3, 9)} colleagues and mentored juniors; "
                           f"{poss} initiatives improved delivery speed by {rng.randint(10, 35)}%.")
        text = (f"{full}\n{role}\nEmail: {email}\nLocation: {city}\n\n"
                f"SUMMARY\n{role} with {years} year(s) of experience. {subj} is known for "
                f"clear communication and dependable delivery.\n\n"
                f"SKILLS\n{', '.join(skills)}\n\n"
                f"EXPERIENCE\n{company} - {role} ({2024 - max(years, 1)}-Present)\n"
                + "\n".join(f"- {b}" for b in bullets)
                + f"\n\nEDUCATION\n{degree}, {college} ({2024 - years - 4})\n")
        resumes.append(Resume(
            id=f"res_{i:04d}", text=text,
            entities={"Name": [full], "Designation": [role], "Skills": skills,
                      "Years of Experience": [str(years)], "Location": [city],
                      "Degree": [degree], "College Name": [college],
                      "Companies worked at": [company], "Email Address": [email]}))
    return resumes


# ------------------------------------------------------------------ dispatcher
def load_resumes(path: str | None) -> tuple[list[Resume], str]:
    """Returns (resumes, source_description)."""
    p = Path(path) if path else None
    if p and p.is_dir():
        r = load_folder(p)
        if r:
            return r, f"{len(r)} .txt resumes from {p}"
    elif p and p.exists():
        r = load_csv(p) if p.suffix.lower() == ".csv" else load_kaggle_json(p)
        if r:
            return r, f"{len(r)} resumes from {p.name}"
    r = make_sample_resumes()
    return r, f"{len(r)} SYNTHETIC sample resumes (dataset file not found)"
