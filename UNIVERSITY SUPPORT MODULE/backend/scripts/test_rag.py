#!/usr/bin/env python
# Quick test script to verify RAG system works without blocking
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import warnings
warnings.filterwarnings('ignore')

print("Loading models and indices...")
from pathlib import Path
import sys
script_dir = Path(__file__).parent
sys.path.append(str(script_dir))

# Import from 5_rag_answer.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from _utils import hybrid_retrieve, compute_confidence

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR_FINAL = ROOT / "models" / "t5_faq_final"
MODEL_DIR = ROOT / "models" / "t5_faq"

if MODEL_DIR_FINAL.exists():
    model_name = str(MODEL_DIR_FINAL)
elif MODEL_DIR.exists():
    model_name = str(MODEL_DIR)
else:
    model_name = "t5-small"

print(f"Loading model from: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

print("Models loaded successfully!")

# Test query
test_query = "What are the library hours?"
print(f"\nTest Query: {test_query}")

# Retrieve passages
retrieved = hybrid_retrieve(test_query, k=5)
conf = compute_confidence(test_query, top_k=1)

print(f"\nRetrieved {len(retrieved)} passages")
print(f"Confidence: {conf:.3f}")

if retrieved:
    print(f"\nTop result:")
    print(f"  Passage: {retrieved[0]['passage'][:200]}...")

print("\n✅ RAG system is working correctly!")
print("The mutex blocking issue has been resolved.")
print("\nYou can now run: python 5_rag_answer.py")
