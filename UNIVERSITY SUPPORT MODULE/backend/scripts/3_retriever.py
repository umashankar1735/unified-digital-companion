# scripts/3_retriever.py
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

def bm25_retrieve(query, k=5):
    toks = query.split()
    top = bm25.get_top_n(toks, passages, n=k)
    return [{"passage": p} for p in top]

def hybrid_retrieve(query, k=5):
    # simple hybrid: union of FAISS and BM25 unique passages then rank by simple heuristic
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

if __name__ == "__main__":
    q = input("Query: ")
    for r in hybrid_retrieve(q, k=5):
        print(" - ", r['passage'][:200])