#!/usr/bin/env python
# Minimal FAISS test to identify the blocking issue
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

print("Step 1: Import faiss...")
import faiss
print("✓ FAISS imported")

print("Step 2: Load index...")
index_path = "../indices/faiss_index.bin"
try:
    index = faiss.read_index(index_path)
    print(f"✓ Index loaded: {index.ntotal} vectors")
except Exception as e:
    print(f"✗ Error loading index: {e}")
    exit(1)

print("Step 3: Test search...")
import numpy as np
query_vec = np.random.rand(1, 384).astype('float32')
D, I = index.search(query_vec, 5)
print(f"✓ Search completed: {len(I[0])} results")

print("\n✅ FAISS is working correctly!")
