"""CLI: build the ChromaDB index.   python ingest.py [--data PATH] [--anonymize]"""
import argparse

import config
from data_loader import load_resumes
from rag_engine import TalentSearch

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=config.DATA_PATH, help="Kaggle JSON, CSV, or folder of .txt")
    ap.add_argument("--anonymize", action="store_true", help="redact identity before embedding")
    args = ap.parse_args()
    resumes, source = load_resumes(args.data)
    print("Loaded", source)
    stats = TalentSearch().index(resumes, anonymize_embeddings=args.anonymize)
    print(f"Indexed {stats['resumes']} resumes as {stats['chunks']} chunks into {config.CHROMA_DIR}")
