# Unified Digital Companion (UDC)

## A-to-Z Project Handbook and Interview Guide

**Repository inspected:** `Final Year - PROJECT`  
**Inspection date:** 21 August 2026  
**Audience:** project author, interviewers, maintainers, and evaluators  
**Evidence basis:** every project-authored source/configuration file; all rows of the CSV, JSONL, and Excel datasets; model/index metadata; runtime smoke tests; all supplied PDFs and the presentation; and a classified audit of generated/vendor artifacts.

> **One-sentence description:** UDC is a two-domain question-answering prototype: a Flask-based Indian legal-awareness retriever built around a fine-tuned InLegalBERT sentence encoder, and a FastAPI-based university-support RAG pipeline using BM25, FAISS, a T5-small generator, confidence-based escalation, and optional cross-encoder reranking.

> **Most important interview caveat:** the repository contains a working pair of backend prototypes and a polished static demonstration frontend, but it does **not** currently contain a working intent router, frontend-to-backend integration, SQL database, real authentication, encrypted audit database, or end-to-end production deployment. The papers describe several of those as the target architecture.

---

# Contents

1. Project Summary
2. What the System Does
3. Implementation Status at a Glance
4. Problem Statement, Goals, and Scope
5. Architecture
6. Technology Stack
7. Repository Structure
8. Prerequisites
9. First-Time Setup
10. Run the Application
11. Configuration and Secrets
12. Frontend Guide
13. Legal Awareness Backend Guide
14. University Support Backend Guide
15. Authentication and Authorization
16. Data Storage and “Database” Guide
17. Dataset Guide
18. Model Training and ML Artifacts
19. Core Business Workflows
20. API Conventions
21. Complete API Reference
22. API Examples
23. Testing and Verification
24. Evaluation Metrics and How to Interpret Them
25. Logging, Auditing, and Error Handling
26. Security, Privacy, Safety, and Legal Currency
27. Troubleshooting
28. Known Gaps and Engineering Risks
29. Recommended Production Architecture
30. Interview Preparation
31. Demo Script
32. Glossary and Quick Reference
33. Repository-Wide File Audit
34. Evidence and Inspection Notes

---

# 1. Project Summary

The project is called **Unified Digital Companion** in the papers and presentation, while the frontend expands UDC as **Universal Digital Campus**. The more accurate name for the research vision is *Unified Digital Companion*: one assistant that routes questions to domain-specific knowledge systems. The current checkout implements two independent domains:

1. **Legal Awareness** - retrieves answers from Indian legal QA datasets. It uses a locally saved BERT-family SentenceTransformer, exact/section-number matching, semantic cosine similarity, Flask routes, and legal disclaimers.
2. **University Support** - answers campus questions from a 503-row Amrita Amaravati knowledge base. It builds BM25 and FAISS indexes, retrieves passages, uses a fine-tuned T5-small generator, and logs low-confidence queries.

The **frontend** is a seven-page HTML/CSS/JavaScript administrative dashboard. It looks complete, but its query, settings, report, evaluation, and logout actions are simulations. No page issues a real `fetch()` request.

There is no relational database. The persistence layer is file-based: CSV, XLSX, JSONL, JSON, FAISS binary indexes, Python pickle, model weights, and an escalation CSV.

## The best honest interview framing

> “We built and evaluated two domain-specific QA backends and a shared administrative UI prototype. The legal service uses a fine-tuned sentence encoder and semantic retrieval; the campus service uses sparse and dense retrieval with a T5 answer generator and a confidence-based human-review path. The checked-in UI is a demonstration layer, and the intent router and production governance components remain research-stage integration work.”

---

# 2. What the System Does

## 2.1 Legal-awareness behavior

A user sends a legal question. The service:

1. Loads `final_merged_legal_qa.csv` and the locally stored fine-tuned BERT encoder.
2. Embeds all 9,763 questions at process startup.
3. Checks for an exact normalized question or an IPC section-number match.
4. Otherwise embeds the query and computes cosine similarity against all question embeddings.
5. Returns the top matches, a formatted answer, category, confidence score, source label, and a criminal-query flag.
6. Appends a legal-information disclaimer when a keyword such as “murder,” “stab,” or “Section 302” is present.

It supports direct question answering, filtered search, category listing, and dataset statistics.

## 2.2 University-support behavior

A user sends a campus question. The service:

1. Loads a BM25 pickle, a FAISS inner-product index, and `meta.json` at import time.
2. Loads `all-MiniLM-L6-v2` for query embeddings.
3. Loads the local `t5_faq_final` T5-small checkpoint if present.
4. Retrieves candidate answers.
5. Computes the top dense-retrieval similarity as confidence.
6. If confidence is below `0.45`, logs the query and returns a domain-specific fallback.
7. Otherwise gives the top passages to T5 and returns the generated answer plus three snippets.

The richer CLI pipeline in `5_rag_answer.py` additionally supports query expansion, optional CrossEncoder reranking, weak-answer detection, and extractive fallback. The FastAPI route does **not** call that richer function; it implements a simpler duplicate path.

## 2.3 Frontend behavior

The frontend provides:

- Dashboard and static metrics.
- Legal and campus query forms.
- Evaluation tables and JSON export.
- Report/download buttons.
- Settings forms.
- Help, FAQ, architecture, and troubleshooting text.

All query results are hard-coded sample records returned after `setTimeout()`. Settings and report actions show alerts but do not change server state or download most claimed artifacts.

---

# 3. Implementation Status at a Glance

| Capability | Status | Repository evidence |
|---|---|---|
| Legal Flask API | Implemented and smoke-tested | `src/app.py`, `src/api/routes.py` |
| Legal semantic retrieval | Implemented | `src/chatbot.py` |
| Legal local model | Present, about 418 MB | `models/fine_tuned_legal_bert/` |
| Legal data preparation/training | Implemented as scripts | `data_preparation.py`, `train_model.py` |
| University FastAPI API | Implemented; runtime answer defect observed | `backend/api/app.py` |
| BM25 and FAISS artifacts | Present | `backend/indices/` |
| Fine-tuned T5-small | Present, about 231 MB | `models/t5_faq_final/` |
| Confidence escalation | Implemented | `api/app.py`, `7_logging.py` |
| CrossEncoder reranking | Implemented in CLI/evaluation, not FastAPI route | `5_rag_answer.py`, `6_rerank.py` |
| Static web UI | Implemented | `frontend/` |
| Frontend/API integration | Not implemented | no `fetch()` in HTML/JS; README calls it pending |
| Intent router | Research claim only | no router source, API, or model in checkout |
| Authentication/RBAC | Not implemented | “Admin” and logout are static/mock |
| SQL database | Not implemented | no SQL/schema/database file |
| Encrypted logging/PII anonymization | Not implemented | plaintext `logs/escalations.csv` |
| Docker support | Partial | Legal Dockerfile omits required data and models |
| Automated test suite | Not implemented | two diagnostic scripts, no assertions/test framework |
| Production readiness | Not established | missing integration, auth, packaging, observability, and valid external evaluation |

---

# 4. Problem Statement, Goals, and Scope

## 4.1 Problem

Legal and institutional knowledge is fragmented across judgments, statutes, university pages, handbooks, circulars, and support contacts. Users often do not know the correct terminology or source. UDC explores whether a unified conversational interface can retrieve grounded domain answers and escalate uncertainty.

## 4.2 Goals implemented in code

- Domain-specific retrieval rather than unrestricted general generation.
- A legal dataset pipeline and local legal encoder.
- Sparse/dense campus retrieval artifacts.
- A lightweight T5-small answer generator.
- Confidence-aware fallback for campus questions.
- A legal disclaimer for sensitive terms.
- Machine-readable evaluation outputs and a demonstration dashboard.

## 4.3 Goals described in research artifacts but not implemented here

- Ensemble intent routing using CNN, BERT, and rules.
- A single unified gateway that chooses the legal or campus service.
- Inline authoritative citations for every generated statement.
- Expert review UI and feedback-driven retraining.
- RLHF/reward-model training.
- Encrypted databases, access control, PII anonymization, and bias audits.
- Kubernetes/scaled deployment and sustained public usage.

## 4.4 Non-goals of the current implementation

- It is not a substitute for professional legal advice.
- It does not access live university systems, AUMS, grades, fees, or personal records.
- It does not update automatically when statutes or campus policies change.
- It does not provide identity, tenant isolation, or permission-aware answers.

---

# 5. Architecture

## 5.1 Architecture that actually exists

```text
                           STATIC DEMONSTRATION UI
                 HTML + CSS + browser JavaScript on :8080
                 (sample data; no API calls implemented)
                              /              \
                             /                \
                            v                  v
              LEGAL SERVICE                    UNIVERSITY SERVICE
              Flask :5000                      FastAPI :8000
              /api/v1/*                        /ask, /escalations
                    |                                  |
        exact/IPC match + cosine search       dense index + BM25 object
                    |                                  |
        fine-tuned BERT encoder               all-MiniLM-L6-v2 encoder
                    |                                  |
        final_merged_legal_qa.csv             FAISS + meta.json + bm25.pkl
                                                       |
                                                local T5-small model
                                                       |
                                             plaintext escalation CSV
```

These are separate processes. There is no gateway or router connecting them.

## 5.2 Legal request sequence

```text
POST /api/v1/ask
  -> validate question exists
  -> lazy-create LegalChatbot
  -> load local SentenceTransformer
  -> read 9,763-row CSV
  -> encode all questions (cold-start cost)
  -> exact/IPC match OR cosine top-k
  -> format answer and optional disclaimer
  -> JSON response
```

## 5.3 University request sequence

```text
process import
  -> load encoder, BM25 pickle, FAISS index, metadata, tokenizer, T5 model

POST /ask
  -> dense-first retrieval function named hybrid_retrieve
  -> compute dense confidence
  -> confidence < 0.45?
       yes -> append CSV escalation -> return fallback
       no  -> concatenate top 3 passages -> T5 generate -> return snippets
```

## 5.4 Important architectural discrepancy: “hybrid” retrieval

`hybrid_retrieve()` requests `k*2` dense results and `k*2` BM25 results, then loops through `dense + bm` and stops as soon as `k` unique passages have been added. Because dense normally supplies at least `k` unique passages, the loop stops before reaching BM25. In normal operation this is **dense-first/dense-only**, not a scored BM25+FAISS fusion.

A genuine hybrid design would normalize both scores and use a weighted sum or reciprocal-rank fusion, or alternate/merge candidates before truncation.

## 5.5 Research target architecture

The papers describe:

```text
User -> intent ensemble -> domain retriever -> cross-encoder reranker
     -> grounded generator -> confidence/risk gate -> answer or expert queue
     -> feedback/audit/retraining loop
```

Treat this as a roadmap, not the current deployment.

---

# 6. Technology Stack

## 6.1 Frontend

- HTML5, CSS3, vanilla JavaScript.
- Roboto loaded from Google Fonts.
- No React/Vue/Angular, build tool, npm package, or bundler.
- Browser APIs: DOM, Blob, object URLs, downloads, printing, confirm/alert.

## 6.2 Legal backend

- Python 3.9 environment.
- Flask 3.0 and Flask-CORS 4.0.
- PyTorch 2.8, Transformers 4.57.6, SentenceTransformers 5.1.2.
- Pandas, NumPy, scikit-learn.
- Local BERT model: 12 layers, 768 hidden size, 12 attention heads, 512-token maximum, mean pooling, cosine similarity.
- Gunicorn listed for deployment.
- FAISS listed but not used in the legal API implementation.

## 6.3 University backend

- Python 3.9 environment.
- FastAPI 0.119, Uvicorn 0.37, Pydantic 2.12.
- SentenceTransformers 5.1.1 and `all-MiniLM-L6-v2` (384-dimensional embeddings).
- FAISS `IndexFlatIP` over normalized passages.
- `rank-bm25` BM25Okapi.
- Transformers 4.57.1, PyTorch 2.8.
- Fine-tuned T5-small: 6 encoder/decoder layers, model width 512, 8 heads.
- Pandas/openpyxl for the source workbook.

## 6.4 Storage

- CSV, JSONL, JSON, XLSX.
- FAISS binary index.
- Pickled BM25 object.
- Safetensors model weights plus tokenizer/config files.
- No SQL engine or ORM.

---

# 7. Repository Structure

```text
Final Year - PROJECT/
├── frontend/                       # seven-page static UI
├── LEGAL AWARENESS MODULE/
│   ├── Ace_Attorney_...Report.pdf
│   └── legal-awareness-backend/
│       ├── src/                    # Flask, retrieval, training, services
│       ├── data/                   # six legal CSV datasets
│       ├── models/                 # local fine-tuned legal encoder
│       ├── Dockerfile
│       ├── requirements.txt
│       ├── run_pipeline.py
│       └── evaluate_system.py
├── UNIVERSITY SUPPORT MODULE/
│   └── backend/
│       ├── api/app.py              # FastAPI service
│       ├── scripts/                # preparation, indexing, RAG, eval
│       ├── data/                   # workbook, CSV, JSONL
│       ├── indices/                # BM25, FAISS, metadata
│       ├── models/                 # T5 checkpoint and final model
│       └── logs/escalations.csv
├── Papers/                         # two papers, report, slide deck
├── evaluate_udc_system.py          # combined evaluation
├── UDC_COMPLETE_EVALUATION_RESULTS.json
└── UDC_PAPER_METRICS.json
```

The checkout is not a Git repository at this root. It also contains copied virtual environments, making it roughly 3.7 GB and 78,000 files.

---

# 8. Prerequisites

## Minimum

- macOS/Linux/Windows with Python 3.9 or a compatible Python version.
- 8 GB RAM minimum; 16 GB is more comfortable because model loading is memory-heavy.
- Several GB of free disk space.
- Internet on the first clean university setup to obtain `all-MiniLM-L6-v2`; the T5 and legal models are already stored locally in this checkout.

## Optional

- Docker for the legal service, after fixing its build context.
- GPU for retraining. Inference works on CPU but cold start is slow.
- A static HTTP server for the frontend.

## Important portability warning

The checked-in virtual environments are not portable. One university `pip` launcher still references an older `.../Datasets/backend/venv/bin/python3` path, although `venv/bin/python` currently runs. Create fresh environments for a clean machine.

---

# 9. First-Time Setup

## 9.1 Legal service

```bash
cd "LEGAL AWARENESS MODULE/legal-awareness-backend"
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The model and `data/final_merged_legal_qa.csv` must remain in their current relative paths.

## 9.2 University service

There is no university `requirements.txt`. Reconstruct the observed environment with:

```bash
cd "UNIVERSITY SUPPORT MODULE/backend"
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  fastapi==0.119.0 uvicorn==0.37.0 pydantic==2.12.2 \
  sentence-transformers==5.1.1 transformers==4.57.1 torch==2.8.0 \
  faiss-cpu==1.12.0 rank-bm25==0.2.2 \
  pandas==2.3.3 numpy==2.0.2 scikit-learn==1.6.1 \
  datasets==4.2.0 accelerate==1.10.1 openpyxl==3.1.5
```

The first SentenceTransformer load may contact Hugging Face even if some cached files exist. For a truly offline deployment, save `all-MiniLM-L6-v2` inside the project and load that local path.

## 9.3 Frontend

No package installation is required.

---

# 10. Run the Application

## Terminal 1 - legal API

```bash
cd "LEGAL AWARENESS MODULE/legal-awareness-backend"
source .venv/bin/activate
python src/app.py
```

Open `http://localhost:5000/` or `http://localhost:5000/api/v1/health`.

## Terminal 2 - university API

Run from the **backend root**, not from `backend/api`, because the module imports `scripts._utils`:

```bash
cd "UNIVERSITY SUPPORT MODULE/backend"
source .venv/bin/activate
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs` for Swagger UI.

## Terminal 3 - static frontend

From the project root:

```bash
python3 -m http.server 8080 --directory frontend
```

Open `http://localhost:8080/`.

> The UI remains a mock even when both APIs are running. Demonstrate backend responses with Swagger or cURL unless the frontend is integrated later.

## Training and rebuilding commands

Legal full pipeline:

```bash
cd "LEGAL AWARENESS MODULE/legal-awareness-backend"
python run_pipeline.py
```

Campus data/index/model pipeline:

```bash
cd "UNIVERSITY SUPPORT MODULE/backend"
python scripts/1_prepare_data.py
python scripts/2_build_indices.py
python scripts/4_train_t5.py
python scripts/5_rag_answer.py
```

These are expensive and may overwrite generated artifacts. Back up models/indexes before retraining.

---

# 11. Configuration and Secrets

## Legal configuration

`src/config.py` defines `SECRET_KEY`, `OPENAI_API_KEY`, `MODEL_PATH`, `DATASET_PATH`, and `DEBUG`, but the active routes use hard-coded repository-relative model/data paths and do not load this class.

`.env.example` defines `API_KEY`, `DATABASE_URL`, `MODEL_PATH`, `DEBUG`, and `SECRET_KEY`. It does not match `OPENAI_API_KEY` or `DATASET_PATH` in `Config`. There is no database client in the code.

`gpt_service.py` is disconnected from the API, hard-codes `/content/fine_tuned_inlegalbert`, expects `OPENAI_API_KEY`, and uses the older `openai.ChatCompletion.create()` interface. `openai` is not listed in requirements. Treat this as dead/experimental code.

## University configuration

Configuration is hard-coded:

- Embedding model: `all-MiniLM-L6-v2`.
- T5 selection: `models/t5_faq_final`, then `models/t5_faq`, then `t5-small`.
- Confidence threshold: `0.45`.
- Query/top-k defaults: 5.
- Threading variables: OMP/MKL/OpenBLAS = 1; duplicate OpenMP allowed.

The frontend settings page does not update these values.

## Secret status

No real secret was found in the project-authored files. Placeholder keys exist. Before publishing, still exclude `.env`, audit logs, copied environments, caches, and model artifacts as appropriate.

---

# 12. Frontend Guide

## 12.1 Shared design

`styles.css` defines a government/academic visual system: dark blue header, navy navigation, light gray background, white content panels, tables, badges, forms, alerts, modal styles, and a mobile breakpoint at 768 px.

`script.js` provides logout confirmation, validation, alerts, modal handling, CSV export, print, a blocking loading overlay, date formatting, and page initialization.

## 12.2 Page-by-page behavior

### `index.html`

Static dashboard with record totals, legal/campus metrics, module status, system information, and navigation.

### `legal-module.html`

Legal query form and a three-row hard-coded Section 420 result. It detects a small browser-side criminal keyword list. It does not call Flask.

### `university-module.html`

Campus query form and three hard-coded library results. It does not call FastAPI. Its Export button references table ID `campusResultsTable`, but the table has no such ID, so export would fail.

### `analysis.html`

Static metric tables. “Run Complete Evaluation” waits three seconds. JSON export builds values in the browser rather than reading `UDC_PAPER_METRICS.json`.

### `reports.html`

Most download buttons only show success alerts. Only the metrics CSV and results JSON are actually constructed for download. Several named artifacts are not present under those names.

### `settings.html`

Forms are visual only. A bug passes the form DOM object to `validateForm()`, which expects an element ID and calls `getElementById`; submitting can therefore throw. The displayed `/query` API paths are wrong.

### `help.html`

Contains user documentation and a research-target architecture diagram. The FastAPI startup command changes into `backend/api`, which can break the `scripts` import; run from `backend` instead.

## 12.3 Frontend integration contract

To integrate later:

- Legal: `POST http://localhost:5000/api/v1/ask` with `{"question":"...","top_k":3}`.
- University: `POST http://localhost:8000/ask` with `{"question":"...","top_k":5}`.
- Add CORS middleware to FastAPI; Flask already enables permissive CORS.
- Map the different response shapes into a common view model.
- Use `textContent`, not unsanitized `innerHTML`, for server-returned text.
- Replace fake status/metrics with health and stats calls.

---

# 13. Legal Awareness Backend Guide

## 13.1 Application factory

`src/app.py` creates Flask, enables CORS for all origins, registers `api_bp` at `/api/v1`, and provides the root service manifest.

## 13.2 Route layer

`src/api/routes.py` lazily creates two global objects:

- `LegalChatbot` for ask/categories/stats.
- `LegalRetrievalSystem` for filtered search.

Each object creates its own chatbot and embeddings, so using both paths can duplicate model memory and startup work.

## 13.3 Retrieval logic

`LegalChatbot.retrieve_answer()`:

- normalizes only terminal `?`/`.` and case;
- scans every row for exact equality;
- recognizes the first number in questions containing both “section” and “ipc”;
- assigns manual confidence 0.99 or 0.95 to those matches;
- otherwise runs cosine similarity and `torch.topk`.

The label “Confidence Score” is therefore a mixture of hard-coded match priority and cosine similarity, not a calibrated probability.

## 13.4 Formatting and safety

High-confidence IPC responses use a direct answer format. Other responses include the matched dataset question. Criminal-keyword queries append a disclaimer. The current text tells users to dial 100; India’s official integrated emergency system uses 112, and existing emergency numbers are integrated into it.

## 13.5 Data preparation

`DataPreparation` loads IPC, IndicLegalQA, Ace Attorney, procedural, and an existing merged file. The existing merged file is detected but deliberately excluded from the new merge. It standardizes to question/answer/category/keywords/source, prioritizes IPC rows, removes duplicate questions, adds a keyword-based safety boolean, and writes the final CSV.

## 13.6 Training

`LegalModelTrainer`:

- stratified 80/20 split by category;
- base `law-ai/InLegalBERT` SentenceTransformer;
- MultipleNegativesRankingLoss over question-answer pairs;
- 3 epochs, batch size 16, 10% warmup;
- saves a SentenceTransformer model.

Its “accuracy” marks retrieval correct when the embedding of a retrieved answer has cosine similarity >= 0.65 to the test answer. This is not exact QA accuracy and can be optimistic for related answers.

## 13.7 Unused scaffolding

Pydantic request/response schemas, `EmbeddingService`, `RetrievalService`, `GPTService`, `Config`, and helper functions are not wired into active routes.

---

# 14. University Support Backend Guide

## 14.1 API startup

`api/app.py` imports retrieval artifacts and models at module import. This makes startup fail if indexes, model files, or network/cache dependencies are unavailable.

The API chooses local `t5_faq_final`, then `t5_faq`, otherwise remote `t5-small`.

## 14.2 Retrieval utilities

`_utils.py` loads:

- SentenceTransformer `all-MiniLM-L6-v2`.
- Pickled BM25 object.
- FAISS index.
- JSON metadata.

Dense embeddings are normalized, so FAISS inner product acts as cosine similarity. BM25 tokenization is a raw case-sensitive `split()` with no punctuation normalization.

## 14.3 Answer generation

The FastAPI route builds one prompt: `Context: ... Question: ...`, truncates to 512 tokens, and uses beam search. During inspection, it retrieved the correct Central Library snippet with confidence `0.7691` but returned an empty `answer`. Unlike `5_rag_answer.py`, the API route has no weak/empty-generation fallback.

## 14.4 Richer CLI pipeline

`5_rag_answer.py` adds:

- query expansion by rough intent keywords;
- optional CrossEncoder reranking;
- top-two context construction;
- low-confidence domain fallbacks;
- weak-answer detection;
- extractive fallback and two hard-coded personnel answers.

This duplication is a maintenance risk. The API should import a single tested service function.

## 14.5 Training

`4_train_t5.py` creates an 80/20 split, prefixes inputs with “Answer this question:”, trains T5-small for 3 epochs with batch size 8, and saves `t5_faq_final`. Checkpoint metadata shows 153 steps and decreasing eval loss from 0.7897 to 0.7124.

Padding token IDs are not replaced with `-100` in labels, so padding can contribute to loss. No answer metric is computed during training.

## 14.6 Logging

Low-confidence questions append UTC timestamp, raw query, confidence, and a retrieved snippet to `logs/escalations.csv`. The API returns the last 100 raw lines through `/escalations` without authentication.

---

# 15. Authentication and Authorization

There is **no authentication** in either backend or the frontend.

- “Admin” is static text.
- Logout only shows a confirmation and alert.
- No login page, session, JWT, cookie, API key validation, OAuth, user table, password hash, role check, or route protection exists.
- `/escalations` exposes potentially sensitive user queries to anyone who can reach the API.

For production: put an identity provider or gateway in front, validate JWT access tokens, use roles such as user/reviewer/admin, restrict logs and maintenance routes, apply CSRF protection if cookie auth is used, and maintain an audit trail.

---

# 16. Data Storage and “Database” Guide

## 16.1 There is no SQL database

No `.sql`, SQLite, PostgreSQL schema, migration, ORM model, or connection code exists. The `.env.example` `DATABASE_URL` and report claims about SQLite/PostgreSQL are aspirational.

## 16.2 Actual stores

| Store | Purpose | Update mechanism |
|---|---|---|
| Legal CSV files | Source and final QA records | Pandas scripts |
| Campus XLSX | Human-authored source knowledge | Excel/manual edit |
| `passages.csv` | Normalized campus indexing input | `1_prepare_data.py` |
| `faq_qa.jsonl` | T5 training input | `1_prepare_data.py` |
| `bm25.pkl` | Serialized sparse retriever and passages | `2_build_indices.py` |
| `faiss_index.bin` | Dense vector index | `2_build_indices.py` |
| `meta.json` | FAISS IDs/questions/passages | `2_build_indices.py` |
| Safetensors/config/tokenizer files | Trained models | training scripts |
| `escalations.csv` | Low-confidence review log | `7_logging.py` |

## 16.3 If SQL is added later

Suggested tables: `users`, `knowledge_documents`, `qa_pairs`, `model_versions`, `queries`, `retrieval_results`, `responses`, `escalations`, `expert_reviews`, and `audit_events`. Store large vectors in a vector-capable service or extension; store model binaries in artifact/object storage, not database rows.

---

# 17. Dataset Guide

## 17.1 Legal data - observed statistics

| File | Rows | Important finding |
|---|---:|---|
| `IndicLegalQA Dataset_10K_Revised.csv` | 10,000 | 299 duplicate questions |
| `ace_attorney_qa_dataset.csv` | 2,000 | 1,976 duplicate questions; fictional game-style dialogue |
| `ipc_qa_50.csv` | 28 | Despite name, only 28 rows, Sections 299-326 |
| `procedural_qa_enhanced.csv` | 10 | Practical legal procedures |
| `merged_qa_dataset.csv` | 12,000 | 2,275 duplicate questions; old intermediate |
| `final_merged_legal_qa.csv` | 9,763 | zero duplicate questions |

Final dataset distribution:

- 9,731 “General Legal”; 32 “Criminal Law”.
- Sources: 9,701 Indic rows, only 24 unique Ace rows, 10 procedural rows, and 28 IPC-section rows.
- 180 rows have `requires_safety_response = true`.
- 9,725 rows have an empty `keywords` field.

The class split is extremely imbalanced. The Ace source is not a real legal authority and should not be used for factual legal production answers.

## 17.2 Campus data

The workbook, CSV, and JSONL contain the same 503 unique questions.

| Category | Rows |
|---|---:|
| IT Facilities & Support | 105 |
| Library & Learning Resources | 103 |
| Sports & Physical Education | 100 |
| Student Council & Student Affairs | 97 |
| Timetables & Calendars | 93 |
| General Campus Information | 5 |

There are no nulls or duplicate questions in the campus dataset. The frontend’s category table (Academic/Hostel/Admission/etc.) does not match these actual categories.

## 17.3 Data lifecycle

```text
Campus_Knowledge_Dataset.xlsx
  -> normalize text and assign ID
  -> passages.csv + faq_qa.jsonl
  -> BM25 pickle + normalized FAISS index + meta.json
  -> T5 train/eval split -> checkpoint -> final model
```

```text
IPC + IndicLegalQA + Ace + procedural CSVs
  -> standardize fields -> concatenate -> exact-question deduplicate
  -> safety keyword flag -> final_merged_legal_qa.csv
  -> SentenceTransformer fine-tuning -> local legal encoder
```

---

# 18. Model Training and ML Artifacts

## 18.1 Legal encoder

- Architecture: BERT encoder, not a generative legal LLM.
- 12 transformer layers, 768-dimensional hidden state, 12 heads.
- Mean-token pooling produces sentence embeddings.
- Cosine similarity is the configured comparison.
- Model weight size: approximately 418 MB.

The name “fine-tuned legal BERT” is fair; “the model generates legal answers” is not. The active legal path retrieves stored answers.

## 18.2 Campus generator

- T5-small encoder-decoder.
- 6 layers on each side, width 512, 8 heads.
- Final weight size: approximately 231 MB.
- Training checkpoint includes a 462 MB optimizer state; it is unnecessary for inference.

## 18.3 Campus index

- 503 normalized 384-dimensional answer embeddings.
- Flat exact inner-product search, not approximate clustering.
- About 756 KB for the FAISS index and 156 KB for BM25.

## 18.4 Model artifact hygiene

Do not ship optimizer state, virtual environments, or both checkpoint/final copies in a production image. Version datasets, model config, metrics, and code commit together. Record data hash, split, random seed, and dependency lock.

---

# 19. Core Business Workflows

## 19.1 Ask a legal question

Input question -> retrieve stored legal QA -> return ranked matches -> append safety disclaimer for criminal keywords.

## 19.2 Search legal records with filters

Input query and optional category/minimum score -> retrieve three semantic matches -> filter those matches. Note that it filters **after** top-three retrieval and does not reformat the answer when the best match changes.

## 19.3 Ask a campus question

Input question -> retrieve candidates -> confidence gate -> fallback/escalation or T5 answer plus source snippets.

## 19.4 Human escalation

The implemented workflow is only CSV logging and a GET endpoint. There is no review UI, assignment, correction, resolution status, notification, or retraining automation.

## 19.5 Rebuild campus knowledge

Edit workbook -> prepare CSV/JSONL -> rebuild both indexes -> optionally retrain T5 -> restart API.

## 19.6 Rebuild legal model

Edit source CSVs -> run full pipeline -> regenerate final CSV -> fine-tune encoder -> restart API. The process is compute-intensive and downloads the base model unless cached.

---

# 20. API Conventions

- JSON request and response bodies.
- No shared versioning: legal is `/api/v1`, university is unversioned.
- No auth headers.
- Legal errors use `{"success": false, "error": "..."}` for 500s, but validation errors omit `success`.
- University exceptions are not normalized into a common error contract.
- Legal health returns 200 without verifying model/data readiness.
- University has autogenerated OpenAPI but no health/readiness route.
- No request ID, pagination, rate limiting, maximum question length, or timeout policy.

---

# 21. Complete API Reference

## 21.1 Legal API - port 5000

### `GET /`

Returns service name, version, status, and endpoint map. Does not load the model.

### `GET /api/v1/health`

Returns `{status, service, version}` with 200. It is a liveness check only.

### `POST /api/v1/ask`

Request:

```json
{"question":"What is Section 302 IPC?","top_k":3}
```

Success 200 fields: `success`, `query`, formatted `response`, `best_match`, `is_criminal_query`, `all_matches`.

Errors: 400 if question is missing; 500 for model/data/top-k/runtime errors.

### `POST /api/v1/search`

Request:

```json
{
  "query":"murder",
  "filters":{"category":"Criminal Law","min_score":0.7}
}
```

Success 200 fields: `success`, `query`, `response`, `best_match`, `results`.

### `GET /api/v1/categories`

Returns category strings and total. Loads the entire chatbot on first use.

### `GET /api/v1/stats`

Returns total QA pairs, category counts, and a count where category contains “Criminal”. Loads the chatbot on first use.

## 21.2 University API - port 8000

### `POST /ask`

Request:

```json
{"question":"Where is the Central Library?","top_k":5}
```

High-confidence response: `answer`, `confidence`, `sources`, `escalated:false`.

Low-confidence response: `answer` fallback, `confidence`, `escalated:true`; no `sources` field.

### `GET /escalations`

Returns up to the last 100 CSV lines as strings. This is sensitive and should be admin-only.

### Framework endpoints

- `GET /docs`
- `GET /redoc`
- `GET /openapi.json`

---

# 22. API Examples

Legal health:

```bash
curl -s http://localhost:5000/api/v1/health
```

Legal question:

```bash
curl -s http://localhost:5000/api/v1/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is Section 302 IPC?","top_k":3}'
```

Legal filtered search:

```bash
curl -s http://localhost:5000/api/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"grievous hurt","filters":{"category":"Criminal","min_score":0.60}}'
```

Campus question:

```bash
curl -s http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"Where is the Central Library located on the Amaravati campus?","top_k":5}'
```

Campus escalation log:

```bash
curl -s http://localhost:8000/escalations
```

---

# 23. Testing and Verification

## 23.1 Inspection performed for this handbook

- All 6,871 authored/config/JSON/HTML/CSS/JS lines were read.
- Every row of all legal CSVs and campus XLSX/CSV/JSONL was parsed and statistically audited.
- All authored Python files passed `py_compile`.
- Legal root: 200.
- Legal health: 200.
- Legal empty ask: 400.
- Real legal Section 302 request: 200 with three ranked matches; cold start about one minute on this Mac.
- Real university import/request: models and indexes loaded; correct library snippet retrieved with 0.7691 confidence; API answer was empty.
- Four PDFs were text-extracted and visually contact-sheeted; the 10-slide deck’s text/media structure and first-slide rendering were inspected.

## 23.2 Existing “tests”

- `test_faiss_minimal.py` prints progress and uses a random vector; it has no assertions and its relative index path depends on current working directory.
- `test_rag.py` prints a single retrieval result; it has no assertions.

Neither is a pytest test suite.

## 23.3 Recommended automated tests

- Unit tests for normalization, safety keywords, exact section matching, retrieval filtering, confidence threshold, and fallbacks.
- Deterministic FAISS/BM25 fixtures with a tiny in-memory corpus.
- Flask and FastAPI API contract tests.
- Empty/malformed JSON, invalid `top_k`, oversized input, model missing, and index missing tests.
- Regression test that `/ask` never returns an empty high-confidence answer.
- Frontend tests that mock API payloads and verify safe rendering.
- Offline-start test with all model paths local.

---

# 24. Evaluation Metrics and How to Interpret Them

## 24.1 Saved retrieval results

Legal baseline reports Recall@1 0.994 and answer accuracy 0.995. Campus baseline reports Recall@1 0.700 and answer accuracy 0.715; reranking raises Recall@10 from 0.972 to 0.982 but lowers reported answer accuracy to 0.695.

## 24.2 Major validity limitations

1. **Self-retrieval:** evaluation questions are sampled from the same indexed corpus, and the exact same question/answer remains searchable. This measures lookup of known items, not generalization.
2. **Legal exact-match shortcut:** exact questions receive a hard-coded 0.99 score.
3. **Grounding is answer-vs-itself:** combined evaluation sets `retrieved_context = predicted_answer`, so semantic grounding approaches 100% by construction.
4. **Hallucination is answer-vs-itself:** the same assignment makes out-of-context token count zero by construction.
5. **Intent routing numbers are hard-coded:** `evaluate_intent_routing()` returns constants; no router predictions are evaluated.
6. **Reranker confidence is an unbounded logit:** saved legal average “confidence” 7.6166 is not a probability.
7. **MRR and MAP are identical:** each query has only one defined relevant answer, so the implementations reduce to reciprocal rank.
8. **Baseline comparison values are embedded constants:** they are not recomputed in this repository.

## 24.3 How to discuss metrics in an interview

Say: “The checked-in evaluation was an internal retrieval sanity benchmark and it revealed strong memorized-item retrieval. For a publication-grade claim, I would split by case/document, keep test questions outside the index, define graded relevance, calibrate confidence, and have experts score correctness, citation faithfulness, and risk.”

## 24.4 Better evaluation design

- Group split legal records by case name to prevent leakage.
- Use paraphrased/unseen questions and temporal holdouts.
- Create multiple relevance judgments per query.
- Report Recall@K, nDCG, MRR, exact match, token F1, citation precision/recall, faithfulness, abstention precision, calibration error, latency percentiles, and failure categories.
- Measure router metrics from a labeled legal/campus/out-of-domain dataset.
- Have legal experts review a stratified risk sample.

---

# 25. Logging, Auditing, and Error Handling

## Current logging

- Flask mainly returns exception strings to clients; no structured server logger.
- FastAPI uses basic INFO logging for model choice.
- Campus escalation uses plaintext CSV.
- There is no rotation, retention, correlation ID, redaction, resolution state, or monitoring dashboard.

## Error behavior

- Legal route broad `except Exception` reveals raw errors.
- Missing/invalid JSON can become a 500.
- Invalid top-k can reach `torch.topk` unchecked.
- University startup failures occur before the API can serve a diagnostic route.
- Reranker loading catches `ImportError` but not network/model-load errors.

## Production recommendation

Use structured JSON logs; request IDs; separate audit/security logs; redaction; safe public errors; detailed private exceptions; metrics for latency, confidence, escalation, empty answer, and model load; and health/readiness endpoints that actually test dependencies.

---

# 26. Security, Privacy, Safety, and Legal Currency

## 26.1 Security

- Permissive Flask CORS.
- No FastAPI CORS for browser integration.
- No auth/rate limit/input size limit.
- Unsanitized `innerHTML` patterns could become XSS once real API text is rendered.
- Deserializing `bm25.pkl` is unsafe if the file can be replaced; pickle can execute code.
- Raw exception text may disclose paths and implementation details.

## 26.2 Privacy

The escalation CSV stores real user queries and retrieved snippets in plaintext, and `/escalations` exposes them. This contradicts the paper’s claims of encrypted databases and ingestion-time PII anonymization.

## 26.3 Legal safety

The legal dataset includes old IPC/CrPC labels and fictional Ace Attorney dialogue. India’s Bharatiya Nyaya Sanhita, 2023 came into force on 1 July 2024 (with the notified Section 106(2) exception), so an interview-safe answer is: “The current corpus is a research snapshot and needs authoritative BNS/BNSS/BSA migration, versioned sources, effective dates, and lawyer review before public use.”

Official references:

- India Code BNS: https://www.indiacode.nic.in/handle/123456789/20062
- Ministry of Home Affairs new criminal laws: https://www.mha.gov.in/en/commoncontent/new-criminal-laws
- Government of India emergency system: https://112.gov.in/

The UI sample also calls Section 420 “theft”; Section 420 IPC concerns cheating/dishonest inducement, not the general theft provision. Do not use static UI answers as legal evidence.

---

# 27. Troubleshooting

## `ModuleNotFoundError: scripts`

Run Uvicorn from `UNIVERSITY SUPPORT MODULE/backend`, not `backend/api`:

```bash
uvicorn api.app:app --port 8000
```

## Hugging Face DNS/network retries

`all-MiniLM-L6-v2` is referenced by name and may contact Hugging Face. Ensure internet/cache access or save the model locally and change the path.

## Legal request takes about a minute on first use

The service recomputes 9,763 embeddings on lazy initialization. Persist embeddings and load them with the model.

## University answer is empty but sources are correct

The FastAPI generator lacks the weak-answer fallback in `5_rag_answer.py`. Centralize the answer service and fall back to the best retrieved passage when decoding is empty.

## Frontend still shows sample answers

Expected: no real API integration exists. Replace `setTimeout()` with fetch calls and add FastAPI CORS.

## Settings form throws an error

`validateForm()` expects a string ID, but settings passes `this`. Change the utility to accept a form element or pass the ID.

## Export campus results fails

The referenced table ID does not exist. Add the matching ID or select the existing table directly.

## Legal Docker container answers fail

The Dockerfile copies only `src/`; it omits `data/` and `models/`. Include runtime artifacts or mount them as read-only volumes. Also test the correct Flask import path.

## `pip` in the copied university venv fails

It contains an absolute shebang from the old location. Recreate the virtual environment instead of editing copied scripts.

## OpenSSL warning

The copied Python 3.9 environment uses LibreSSL while urllib3 v2 expects OpenSSL 1.1.1+. Use a current Python distribution and rebuild the environment.

---

# 28. Known Gaps and Engineering Risks

| Severity | Risk | Impact |
|---|---|---|
| Critical | Legal corpus is not current authoritative law | Unsafe or obsolete legal answers |
| Critical | Public escalation endpoint and plaintext logs | Privacy/data exposure |
| Critical | Evaluation leakage and tautological grounding | Unsupported performance claims |
| High | Frontend is not integrated | Demo does not prove end-to-end system |
| High | Intent router absent | “Unified/multi-agent” claim not implemented |
| High | Campus API can return empty answer | Visible functional failure |
| High | “Hybrid” merge truncates before BM25 | Architecture claim differs from behavior |
| High | No authentication or authorization | No safe multi-user/admin deployment |
| High | Docker image omits models/data | Containerized ask route cannot work as built |
| High | Startup downloads and eager loads | Offline/reliability and cold-start failure |
| Medium | Duplicated RAG logic | Drift between CLI and API behavior |
| Medium | No university dependency manifest | Setup is not reproducible |
| Medium | Hard-coded paths/config/contacts | Environment-specific behavior |
| Medium | No persisted legal embeddings | Approximately one-minute cold start observed |
| Medium | Uncalibrated “confidence” | Misleading safety decisions |
| Medium | Copied venvs/checkpoints | 3.7 GB repository, poor portability |
| Medium | No real automated tests | Regressions are likely |
| Medium | XSS-prone rendering patterns | Unsafe after API integration |
| Low | Naming conflict: Unified vs Universal | Interview and branding confusion |
| Low | Paper/deck/report metrics conflict | Credibility/reproducibility concern |

---

# 29. Recommended Production Architecture

1. Build a small gateway API with a real labeled intent classifier and explicit out-of-domain state.
2. Create one service layer per domain; keep framework routes thin.
3. Persist embeddings and version indexes.
4. Implement real reciprocal-rank fusion or score-normalized hybrid retrieval.
5. Make retrieval-first extractive answers the safe default; generate only with citation checking.
6. Add auth/RBAC and protect all logs/admin operations.
7. Store structured records in PostgreSQL; use object storage for models and a supported vector store/index for embeddings.
8. Add a review queue with states, assignee, correction, reason, and resolution.
9. Version law/policy documents by source, jurisdiction, publication date, and effective date.
10. Integrate the frontend through one stable, versioned API response model.
11. Containerize each service with pinned dependencies, health/readiness checks, and artifact mounts.
12. Add CI for linting, unit/API tests, offline startup, model contract tests, and evaluation leakage checks.

---

# 30. Interview Preparation

## 30.1 30-second pitch

“UDC is a dual-domain question-answering prototype for legal awareness and campus support. I used domain-specific retrieval so answers come from controlled datasets rather than unconstrained generation. The legal service uses a fine-tuned InLegalBERT sentence encoder and Flask. The campus service combines a FAISS semantic index, a BM25 index, T5-small generation, and confidence-based escalation through FastAPI. I also built a shared admin UI prototype and evaluation tooling. The next production step is integrating the UI, implementing the router and authentication, and replacing the internal self-retrieval benchmark with leakage-free expert evaluation.”

## 30.2 Two-minute architecture answer

Explain the two services separately, then the research target. Say that legal is retrieval-based, campus is retrieval plus generation, and the static UI is not yet connected. Describe files as the current persistence layer. Finish with confidence fallback and the production roadmap.

## 30.3 Likely questions and strong answers

### Why use RAG/retrieval?

It constrains answers to a maintained corpus, supports source snippets, enables updates without retraining every fact, and is safer than an unrestricted generator in legal/policy domains.

### Why both BM25 and FAISS?

BM25 is strong for exact terms and section numbers; FAISS semantic embeddings handle paraphrases. In the current code the merge needs correction because dense candidates are truncated before BM25 contributes.

### What does the cross-encoder do?

It jointly reads a query-passage pair and gives a stronger relevance score than independent embeddings, at higher latency. It exists in CLI/evaluation code but is not used by the current FastAPI route.

### Why T5-small?

It is a manageable encoder-decoder for controlled sequence-to-sequence fine-tuning and CPU-friendly experimentation. The tradeoff is limited reasoning and occasional weak/empty decoding.

### What is InLegalBERT doing?

It is used through SentenceTransformers as a legal-domain sentence encoder. It converts questions to embeddings for similarity retrieval; it does not generate the final legal prose.

### How is confidence calculated?

Campus confidence is the top normalized dense inner-product score. Legal exact/section matches use manual 0.99/0.95 values; semantic results use cosine similarity. These are ranking scores, not calibrated probabilities.

### What happens below the threshold?

Campus queries below 0.45 are logged and receive a safe hard-coded fallback directing users to the relevant office. A real expert-review lifecycle is still to be implemented.

### How did you evaluate?

The repository has internal retrieval sanity evaluations. Be explicit that many test questions remain in the search index, so results are optimistic. Propose document-level splits and human faithfulness review.

### Is it production ready?

No. Say: “The backend components are functional prototypes; production requires current authoritative datasets, integrated UI, auth, privacy controls, packaging, monitoring, and rigorous evaluation.”

### Where is SQL?

There is none in this implementation. CSV/JSONL plus FAISS/BM25 artifacts were sufficient for the prototype. PostgreSQL would be the next step for users, queries, reviews, and audit state.

### What was the hardest engineering problem?

A defensible answer: aligning data schemas and retrieving relevant domain answers while managing model/index startup costs and uncertainty. Mention discovering that retrieval quality and evaluation design matter more than simply adding a larger generator.

### What would you improve first?

1. Fix the university empty-answer path and centralize service logic.
2. Persist legal embeddings.
3. Implement true hybrid fusion.
4. Integrate the UI.
5. Correct evaluation leakage.
6. Migrate legal sources to current law with provenance.

## 30.4 Claims to avoid

Do not say the repository currently has:

- a working multi-agent intent router;
- 15,000 real institutional support logs;
- encrypted database storage;
- deployed Kubernetes services;
- production authentication;
- verified zero hallucination;
- a fully integrated frontend;
- live nationwide/campus usage proven by this checkout.

## 30.5 Positive but accurate differentiators

- Two distinct domain pipelines in one research program.
- Local trained model artifacts are present, not just pseudocode.
- Dataset and index-building pipelines are reproducible in source.
- Confidence-based abstention is represented in code.
- The project includes both retrieval evaluation and a UI prototype.
- The inspection identified clear next steps—a good engineering maturity signal.

---

# 31. Demo Script

## Before the interview

1. Start the legal API and wait for the first request to finish embedding.
2. Start the university API from the backend root with network/cache available.
3. Open Flask health and FastAPI Swagger.
4. Open the static frontend only as a UI prototype.
5. Keep terminal output and this handbook available.

## Five-minute demo

1. **Problem (30 sec):** explain fragmented legal/campus knowledge.
2. **Architecture (45 sec):** show the two backends and file-based stores.
3. **Legal live call (60 sec):** ask Section 302; point to exact match, ranked alternatives, category, source label, and disclaimer. State the corpus is an IPC-era research snapshot.
4. **Campus live call (60 sec):** use Swagger to ask a known campus question. If the current API returns an empty answer, show that retrieval sources are correct and explain the known generator fallback defect rather than hiding it.
5. **Data pipeline (45 sec):** show the 503-row workbook -> CSV/JSONL -> indexes and the 9,763-row legal merge.
6. **UI (30 sec):** show the consistent pages; explicitly call it a static prototype pending API integration.
7. **Evaluation/roadmap (30 sec):** acknowledge internal benchmark leakage and describe the stronger split and governance plan.

## Backup if models fail

Show saved JSON results, model configs, dataset statistics, and architecture. Never claim a simulated browser response came from the backend.

---

# 32. Glossary and Quick Reference

| Term | Meaning in this project |
|---|---|
| RAG | Retrieve evidence before generating/returning an answer |
| BM25 | Sparse lexical ranking based on token occurrence |
| FAISS | Vector similarity index library |
| Dense retrieval | Semantic search using learned embeddings |
| Cross-encoder | More expensive joint query-passage relevance model |
| InLegalBERT | Legal-domain BERT used as the base legal encoder |
| SentenceTransformer | Wrapper producing fixed-size text embeddings |
| T5-small | Encoder-decoder used for campus answer generation |
| Top-K | Number of results requested/returned |
| Cosine similarity | Directional similarity between embeddings |
| Confidence threshold | Project cutoff `0.45` for campus escalation |
| HITL | Human-in-the-loop review; only logging is implemented |
| EM | Exact string match after normalization |
| F1 | Token-overlap harmonic mean of precision/recall |
| Recall@K | Whether a relevant item is present in first K results |
| MRR | Mean reciprocal rank of first relevant result |
| MAP | Mean average precision; simplified here to one relevant item |
| Grounding | Whether answer content is supported by retrieved evidence |
| Hallucination | Unsupported generated content |
| Liveness | Process responds |
| Readiness | Process and dependencies can serve real requests |

## Port and path reference

| Component | Port/path |
|---|---|
| Frontend | 8080 (suggested static server) |
| Legal API | 5000 |
| Campus API | 8000 |
| Legal question | `/api/v1/ask` |
| Campus question | `/ask` |
| Campus Swagger | `/docs` |
| Confidence cutoff | 0.45 |
| Legal rows | 9,763 |
| Campus rows/vectors | 503 |

---

# 33. Repository-Wide File Audit

## 33.1 Root and frontend

| File/group | Audit result |
|---|---|
| `.vscode/settings.json` | Empty object |
| `.DS_Store` files | macOS metadata; not project logic |
| `evaluate_udc_system.py` | Combined metrics; grounding/hallucination tautology and hard-coded router values |
| `UDC_COMPLETE_EVALUATION_RESULTS.json` | Saved legal/campus retrieval metrics |
| `UDC_PAPER_METRICS.json` | Saved combined metrics dated 2026-02-28 |
| `frontend/README.md` | Accurate that backend integration is pending; example URLs are wrong |
| `frontend/index.html` | Static dashboard |
| `frontend/legal-module.html` | Simulated legal query |
| `frontend/university-module.html` | Simulated campus query; missing export table ID |
| `frontend/analysis.html` | Static metrics and browser-created JSON |
| `frontend/reports.html` | Mostly simulated downloads |
| `frontend/settings.html` | Simulated settings; validation/API-path defects |
| `frontend/help.html` | User/research documentation; some startup and architecture claims mismatch code |
| `frontend/script.js` | Shared DOM utilities; no network client |
| `frontend/styles.css` | Complete shared responsive stylesheet |

## 33.2 Legal backend source

| File | Responsibility/status |
|---|---|
| `.env.example` | Placeholder variables; names do not fully match Config |
| `Dockerfile` | Flask image skeleton; omits runtime data/models |
| `requirements.txt` | Legal dependencies; broad ML lower bounds reduce reproducibility |
| `run_pipeline.py` | Data preparation + legal encoder training orchestration |
| `evaluate_system.py` | Self-retrieval Recall/MRR/MAP benchmark |
| `evaluation_results_legal.json` | Saved benchmark values |
| `src/app.py` | Flask factory, CORS, blueprint, root manifest |
| `src/api/routes.py` | Six active API endpoints and lazy globals |
| `src/api/__init__.py` | Defines a separate unused blueprint |
| `src/chatbot.py` | Main active legal retrieval and formatting logic |
| `src/config.py` | Unused environment-backed Config class |
| `src/data_preparation.py` | Legal merge/dedup/safety pipeline |
| `src/train_model.py` | SentenceTransformer fine-tuning/evaluation |
| `src/schemas/request.py` | Unused Pydantic request models |
| `src/schemas/response.py` | Unused Pydantic response models |
| `src/services/embedding_service.py` | Unused embedding wrapper |
| `src/services/retrieval_service.py` | Unused simpler retrieval wrapper |
| `src/services/gpt_service.py` | Unused/outdated experimental OpenAI path |
| `src/utils/helpers.py` | Unused helper functions |
| package `__init__.py` files | Empty markers |

## 33.3 Legal data/model/report

| File/group | Audit result |
|---|---|
| Six legal CSVs | All rows parsed; statistics in Section 17 |
| `models/fine_tuned_legal_bert/` | Valid SentenceTransformer layout, config/tokenizer/vocabulary/418 MB weights |
| `checkpoints/` | Empty |
| `Ace_Attorney_Performance_and_Comparison_Report.pdf` | Four-page external-framework summary; not evidence that this code implements Ace-Attorney SGP/distillation |
| `source/` and `venv/` | Copied Python environments/vendor code, about 824 MB total; classified rather than treated as authored project logic |

## 33.4 University backend

| File | Responsibility/status |
|---|---|
| `api/app.py` | FastAPI ask/escalations; duplicated simpler generation path |
| `scripts/1_prepare_data.py` | Workbook normalization to CSV/JSONL |
| `scripts/2_build_indices.py` | BM25 and normalized FAISS build |
| `scripts/3_retriever.py` | CLI retrieval; same dense-first merge issue |
| `scripts/4_train_t5.py` | T5-small training |
| `scripts/5_rag_answer.py` | Richest campus inference/CLI path |
| `scripts/6_rerank.py` | CrossEncoder ranking demo |
| `scripts/7_logging.py` | CSV escalation append |
| `scripts/_utils.py` | Shared index loading/retrieval/confidence |
| `scripts/evaluate_system.py` | Campus self-retrieval evaluation |
| `scripts/test_faiss_minimal.py` | Print-based diagnostic, cwd-sensitive path |
| `scripts/test_rag.py` | Print-based single-query diagnostic |
| `scripts/run_rag.sh` | Thread-limited CLI wrapper |
| `evaluation_results_university.json` | Saved retrieval metrics |
| `data/Campus_Knowledge_Dataset.xlsx` | 503-row source workbook, one sheet |
| `data/passages.csv` | Exact normalized 503-row index corpus |
| `data/faq_qa.jsonl` | Exact 503-row training representation |
| `indices/bm25.pkl` | Pickled BM25/passages/IDs |
| `indices/faiss_index.bin` | 503-vector dense index |
| `indices/meta.json` | IDs/questions/passages |
| `logs/escalations.csv` | Plaintext header plus ten observed query records |
| `models/t5_faq/checkpoint-153/` | Training checkpoint including optimizer state |
| `models/t5_faq_final/` | Inference T5 model |
| `venv/`, `scripts/venv/`, `scripts/source/` | Three copied/vendor environments, roughly 2.46 GB together |

## 33.5 Papers and presentation

| File | Audit result |
|---|---|
| `Papers/CRC_587.pdf` | 12-page paper, 82.3% grounding/62.4% F1 research narrative |
| `Papers/UDC_IEE.pdf` | 12-page IEEE-style expanded paper with architecture, ablation, ethics claims |
| `Papers/UDC_Project-Report.pdf` | 61-page report; contains template residue, conflicting metrics/configs, and aspirational SQLite/PostgreSQL/Streamlit claims |
| `Papers/Conference.pptx` | 10-slide conference deck; repeats research-target claims; first slide has visible title/metadata overlap |

## 33.6 Binary/generated inventory

The repository contains approximately 78,270 files. About 78,000 are dependency/environment files. Large artifacts include a 462 MB optimizer state, 418 MB legal weights, two 231 MB T5 model copies, native libraries, tokenizers, indexes, and rendered/document binaries. These were inspected by type, metadata, configuration, size, and role; compiled weights/libraries are not line-oriented source code.

---

# 34. Evidence and Inspection Notes

## Current-source truth order

When sources conflict, use this order:

1. Runtime-observed code behavior.
2. Current source and model/index metadata.
3. Current datasets and saved result JSON.
4. Frontend README.
5. Papers, project report, and slides.

## Confirmed observations

- No Git repository at this root.
- No SQL/database implementation.
- No frontend fetch call.
- No intent-router source.
- No authentication.
- Legal lightweight and full ask routes run.
- Legal cold start recomputes all question embeddings.
- Campus retrieval loads and finds the correct known passage.
- Campus API high-confidence generator returned an empty answer during smoke test.
- All authored Python files are syntactically valid.
- Dataset counts and duplicates were recomputed, not copied from UI labels.

## Final interview principle

The strongest presentation is not to defend every sentence in the older report. Show that you understand what the code actually does, why its design choices were made, what the evaluation can and cannot prove, and how you would turn the prototype into a safe production system.

