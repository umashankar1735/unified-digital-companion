# scripts/6_rerank.py
from sentence_transformers import CrossEncoder
from scripts._utils import hybrid_retrieve
from pathlib import Path

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")  # small & good
def rerank(question, candidates):
    pairs = [[question, c['passage']] for c in candidates]
    scores = model.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [r[0] for r in ranked]

if __name__ == "__main__":
    q = input("Query: ")
    cands = hybrid_retrieve(q, k=10)
    ranked = rerank(q, cands)[:5]
    for r in ranked:
        print("Score snippet:", r['passage'][:200])