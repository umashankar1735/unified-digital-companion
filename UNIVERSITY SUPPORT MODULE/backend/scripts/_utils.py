# scripts/_utils.py
import os
# Set threading environment variables BEFORE importing ML libraries
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pickle, json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "indices"
model = SentenceTransformer("all-MiniLM-L6-v2")
bm25_obj = pickle.load(open(OUT / "bm25.pkl", "rb"))
bm25 = bm25_obj["bm25"]
passages = bm25_obj["passages"]
ids = bm25_obj["ids"]

faiss_index = faiss.read_index(str(OUT / "faiss_index.bin"))
meta = json.load(open(OUT / "meta.json", "r", encoding="utf-8"))

def dense_retrieve(query, k=5):
    q_emb = model.encode([query], convert_to_numpy=True)
    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-10)
    D, I = faiss_index.search(q_emb.astype('float32'), k)
    results = []
    for idx in I[0]:
        results.append({"id": meta["ids"][idx], "passage": meta["passages"][idx]})
    return results



def compute_confidence(query, top_k=1):
    q_emb = model.encode([query], convert_to_numpy=True)
    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-10)
    D, I = faiss_index.search(q_emb.astype('float32'), top_k)
    # D are inner product scores (since faiss index normalized)
    if len(D[0]) == 0:
        return 0.0
    return float(D[0][0])  # top-1 similarity

def bm25_retrieve(query, k=5):
    toks = query.split()
    top = bm25.get_top_n(toks, passages, n=k)
    return [{"passage": p} for p in top]

def hybrid_retrieve(query, k=5):
    dense = dense_retrieve(query, k=k*2)
    bm = bm25_retrieve(query, k=k*2)
    seen = set()
    merged = []
    for item in dense + bm:
        passage = item["passage"]
        if passage not in seen:
            merged.append(item)
            seen.add(passage)
        if len(merged) >= k:
            break
    return merged