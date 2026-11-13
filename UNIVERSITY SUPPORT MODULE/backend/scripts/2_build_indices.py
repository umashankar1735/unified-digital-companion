# scripts/2_build_indices.py
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "passages.csv"
OUT = ROOT / "indices"
OUT.mkdir(exist_ok=True)

# Check if data file exists
if not DATA.exists():
    raise FileNotFoundError(f"Data file not found: {DATA}")

df = pd.read_csv(DATA)

# Validate required columns
required_columns = ['Answer', 'Question', 'id']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

passages = df['Answer'].astype(str).tolist()
questions = df['Question'].astype(str).tolist()
ids = df['id'].astype(str).tolist()

print(f"Loaded {len(passages)} passages from {DATA}")

# BM25
print("Building BM25 index...")
tokenized = [p.split() for p in passages]
bm25 = BM25Okapi(tokenized)
with open(OUT / "bm25.pkl", "wb") as f:
    pickle.dump({"bm25": bm25, "passages": passages, "ids": ids}, f)
print("Saved BM25 index")

# Dense embeddings
print("Loading sentence transformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")  # small & fast
print("Encoding passages...")
embeddings = model.encode(passages, show_progress_bar=True, convert_to_numpy=True)

# Normalize embeddings for cosine similarity
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings = embeddings / (norms + 1e-10)

d = embeddings.shape[1]
index = faiss.IndexFlatIP(d)  # inner-product with normalized vectors = cosine
index.add(embeddings.astype('float32'))
faiss.write_index(index, str(OUT / "faiss_index.bin"))

# Save embedding metadata
meta = {"ids": ids, "questions": questions, "passages": passages}
with open(OUT / "meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False)
print("Saved FAISS index and meta")