#!/bin/bash
# Wrapper script to run RAG answer with proper threading configuration

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE

# Run the Python script
python 5_rag_answer.py
