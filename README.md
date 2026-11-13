# Unified Digital Companion (UDC)
### *A Scalable AI Framework for Legal Awareness and Contextual Public Support*

[![Accepted at Confluence 2026](https://img.shields.io/badge/Publication-Confluence_2026_(Springer_LNEE)-0ea5e9?style=for-the-badge&logo=springer)](Papers/CRC_587.pdf)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-emerald?style=for-the-badge)](LICENSE)

> **Accepted for publication at the 15th International Conference on Cloud Computing, Data Science & Engineering (Confluence 2026), to be published in *Lecture Notes in Electrical Engineering (LNEE)* by Springer Nature.**

---

## 📌 Overview

The **Unified Digital Companion (UDC)** is a multi-agent framework utilizing Retrieval-Augmented Generation (RAG) to provide verified legal awareness, procedural guidance, and institutional support. It addresses the challenge of dispersed, complex statutory information by combining fine-tuned domain-adapted encoders, hybrid dense-sparse vector retrieval, and cross-encoder reranking under zero-hallucination verification gates.

The prototype implements two core domain modules:
1. **Legal Awareness Module**: Retrieves high-confidence statutory answers from Indian penal codes and legal QA corpora using fine-tuned `InLegalBERT`, exact section-number resolution, and cosine similarity.
2. **University Support Module**: Fast RAG pipeline answering institutional documentation, compliance, and academic queries via hybrid `BM25` + dense `FAISS` vector search, a fine-tuned `T5` generation model, and confidence-based human-in-the-loop escalation.

---

## 📊 Evaluation & Benchmark Telemetry

Evaluated against the **IndicLegalQA** benchmark and institutional corpora:

| Metric | UDC System | Naive RAG Baseline | Improvement |
| :--- | :---: | :---: | :---: |
| **Grounding Accuracy** | **82.3%** | 51.2% | **+31.1%** |
| **Intent Routing Accuracy** | **94.1%** | 68.4% | **+25.7%** |
| **F1 Score** | **62.4%** | 44.1% | **+18.3%** |
| **Hallucination Rate** | **0.0%** | 18.6% | **Zero Hallucination** |
| **Average End-to-End Latency** | **< 530 ms** | 1,420 ms | **2.6× Faster** |

---

## 🏗️ Architecture

```
User Query / Statute Inquiry
         │
         ▼
┌─────────────────────────────────┐
│   Multi-Stage Intent Router     │  <-- Ensemble BERT + CNN (94.1% Routing)
└──────────────┬──────────────────┘
               │
      ┌────────┴────────┐
      ▼                 ▼
[Legal Domain]    [University Domain]
      │                 │
      ├─────────────────┼────────────────────────────────────────┐
      ▼                 ▼                                        ▼
┌──────────────┐ ┌──────────────┐                         ┌──────────────┐
│  InLegalBERT │ │     BM25     │                         │  FAISS Index │
│ Dense Vector │ │Sparse Lexical│                         │ Dense Vector │
└──────┬───────┘ └──────┬───────┘                         └──────┬───────┘
       │                │                                        │
       └───────┬────────┴────────────────────────────────────────┘
               ▼
┌─────────────────────────────────┐
│  Reciprocal Rank Fusion (RRF)   │
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│ Cross-Encoder Reranker (MiniLM) │  <-- Joint attention scoring (Top-3)
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│  Grounded T5 Generator + Gate   │  <-- 0.0% Hallucination Gate
└──────────────┬──────────────────┘
               ▼
Verified Answer with Statutory Citation
```

---

## 👥 Authors & Affiliation

* **S. Gani Lakshmi** — *School of Computing, Amrita Vishwa Vidyapeetham, Amaravati*
* **S. Purva Himaja** — *School of Computing, Amrita Vishwa Vidyapeetham, Amaravati*
* **K. Sai Uma Shankar** (`av.en.u4cse22119@av.students.amrita.edu`) — *Amrita Vishwa Vidyapeetham*
* **P. Mohan Reddy** — *School of Computing, Amrita Vishwa Vidyapeetham, Amaravati*
* **P. Kushal Shiva Charith Reddy** — *School of Computing, Amrita Vishwa Vidyapeetham, Amaravati*

---

## 📖 Citation

```bibtex
@inproceedings{lakshmi2026unified,
  title     = {Unified Digital Companion: A Scalable AI Framework for Legal Awareness and Contextual Public Support},
  author    = {Lakshmi, S. Gani and Himaja, S. Purva and Shankar, K. Sai Uma and Reddy, P. Mohan and Reddy, P. Kushal Shiva Charith},
  booktitle = {15th International Conference on Cloud Computing, Data Science and Engineering (Confluence)},
  series    = {Lecture Notes in Electrical Engineering (LNEE)},
  publisher = {Springer Nature},
  year      = {2026},
  address   = {Amaravati, India}
}
```

---

## 🚀 Quickstart

### 1. Legal Awareness Backend (Flask)
```bash
cd "LEGAL AWARENESS MODULE/legal-awareness-backend"
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py
```

### 2. University Support Backend (FastAPI)
```bash
cd "UNIVERSITY SUPPORT MODULE/backend"
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 3. Evaluation Suite
```bash
python evaluate_udc_system.py
```

---

## 📄 License
MIT License. Developed as an academic capstone and published research framework.
