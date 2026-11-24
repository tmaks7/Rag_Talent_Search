"""Central settings for the RAG talent-search engine."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_PATH = os.getenv("RESUME_DATA", str(DATA_DIR / "Entity Recognition in Resumes.json"))

CHROMA_DIR = str(BASE_DIR / "chroma_db")
COLLECTION = "resumes"

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

CHUNK_SIZE = 900        # characters per chunk
CHUNK_OVERLAP = 150
CANDIDATE_POOL = 60     # candidates considered before re-ranking
TOP_K = 3               # resumes sent to the LLM for the fit summary
MAX_RESUME_CHARS = 3500 # per resume inside the LLM prompt
