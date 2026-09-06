# 📄 Agentic CV Question Answering System

A production-oriented agentic AI system that answers questions about candidate CVs, strictly grounded in the CV content. Built with **LangGraph**, **Google Gemini**, and **FAISS**.

Supports **multi-candidate CV indexing**, **cross-candidate comparative reasoning**, **multi-turn context resolution**, and **defense-in-depth grounding verification**.

---

## ✨ Key Capabilities

| Feature | Description |
| :--- | :--- |
| **Agent Architecture** | Multi-node **LangGraph** workflow (Contextualizer → Mandatory Retrieval → Reasoner → Grounding Validator) |
| **LLM Model** | Google Gemini (`gemini-2.5-flash-lite` / configurable) |
| **Embeddings** | Google Gemini Embeddings (`gemini-embedding-001`, 3072 dimensions) |
| **Vector Store** | Local **FAISS** index with multi-file SHA-256 freshness hashing |
| **Document Ingestion** | Full support for `.pdf` and `.docx` with best-effort section inference and heuristic candidate name extraction |
| **Multi-Candidate Support** | Upload and index multiple CVs simultaneously with automatic cross-candidate comparisons |
| **Duration Standardization** | Standardizes tenure and durations in **years** by default (`35 months` → `~2.9 years`), supporting numeric threshold comparisons |
| **Anti-Hallucination Defense** | Mandatory retrieval gate + vector distance thresholding + graph fallback short-circuit + fail-closed secondary LLM grounding validator |
| **Conversational Memory** | Stateful multi-turn memory with pronoun resolution and recruitment abbreviation expansion |
| **Interfaces** | Interactive **Terminal CLI** (`cli.py`) + Modern **Streamlit Web UI** (`app.py`) |
| **Automated Tests** | 24 unit and integration tests passing (`pytest tests/ -v`) |

---

## 🏗️ System Architecture & Workflow

```
User Query (CLI / Streamlit)
           │
           ▼
┌──────────────────────────────────────────┐
│        ConversationMemory                │  <-- Multi-turn sliding window
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│              LangGraph Agentic State Graph                  │
│                                                             │
│  1. [contextualize_query_node]                              │  <-- Rewrites pronouns & abbreviations (e.g. 'yoe', 'it', 'what about X')
│            │                                                │
│            ▼                                                │
│  2. [mandatory_retrieval_node]                              │  <-- ALWAYS calls search_cv before generation (enforced at graph level)
│            │                                                │
│       ┌────┴────────────────────────┐                       │
│       │ (relevant content found)    │ (nothing found)       │
│       ▼                             ▼                       │
│  3. [agent_node]               [fallback_node]              │  <-- Reasoner LLM OR short-circuit
│       │                             │                       │
│  ┌────┴──────────────────┐          │                       │
│  │(additional tool call) │(answer)  │                       │
│  ▼                       │          │                       │
│  4. [tool_node (search_cv)]         │                       │  <-- FAISS similarity search + score filtering (L2 ≤ 1.0)
│       │                             │                       │
│       ▼                             │                       │
│  5. [after_tools_routing]           │                       │
│       │ Chunks relevant?            │                       │
│       ├── NO ─────────┐             │                       │
│       │ YES           │             │                       │
│       ▼               ▼             │                       │
│  6. [agent_node]  [fallback_node]   │                       │  <-- Synthesizes grounded answer or short-circuits
│       │               │             │                       │
│       └───────┬───────┘             │                       │
│               │                     │                       │
│               ▼                     │                       │
│  7. [check_grounding_node] ◄────────┘                       │  <-- Fail-closed secondary LLM validator
│               │                                             │
└───────────────┼─────────────────────────────────────────────┘
                ▼
        Final Grounded Answer
```

---

## 📁 Repository Structure

```
cv_agent/
├── cv_agent/
│   ├── __init__.py          # CVAgent facade (manages lifecycle, memory, graph)
│   ├── config.py            # Pydantic Settings & environment configuration
│   ├── logging_setup.py     # Clean, structured console & file logging
│   ├── ingestion/
│   │   └── loader.py        # PDF & DOCX loaders, chunking, best-effort section inference & candidate name extraction
│   ├── indexing/
│   │   └── vector_store.py  # FAISS index builder, multi-CV SHA-256 hashing, caching
│   ├── tools/
│   │   └── cv_search.py     # @tool search_cv with L2 threshold filtering & excerpt formatting
│   └── agent/
│       ├── state.py         # AgentState TypedDict schema
│       ├── memory.py        # ConversationMemory state manager
│       ├── nodes.py         # Graph node implementations & system prompts
│       ├── graph.py         # LangGraph StateGraph definition & compilation
│       └── session_writer.py# Audit trail session exporter (.json / .txt)
├── cli.py                   # Terminal CLI entry point
├── app.py                   # Streamlit Web UI entry point
├── tests/
│   ├── test_ingestion.py    # Tests for PDF/DOCX loaders & chunking
│   └── test_agent.py        # Tests for memory, contextualization, grounding, fallback
├── data/                    # Candidate CV files (.pdf, .docx) & persistent FAISS index
├── output/                  # Audit logs (agent.log) and saved session transcripts
├── pyproject.toml           # Project dependencies and packaging metadata
├── .env.example             # Environment variable template
└── README.md                # Documentation
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites

- Python 3.10 or higher
- A Google Gemini API key ([Get an API key here](https://aistudio.google.com/app/apikey))

### 2. Environment Setup

```bash
# Clone or navigate to the repository
cd cv_agent

# Create and activate a virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies in editable mode
pip install -e .
```

### 3. Configure `.env`

Copy the template and insert your `GOOGLE_API_KEY`:

```bash
copy .env.example .env    # Windows
# or
cp .env.example .env      # macOS/Linux
```

Ensure your `.env` contains:
```env
GOOGLE_API_KEY=AIzaSy...
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.5-flash-lite
EMBEDDING_MODEL=gemini-embedding-001
RETRIEVAL_TOP_K=6
RETRIEVAL_SCORE_THRESHOLD=1.0
```

---

## 💻 Running the Application

### Option A: Modern Streamlit Web UI

```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

- **Multiple CV Upload**: Upload multiple candidate resumes (`.pdf` or `.docx`) at once.
- **Dynamic Candidate Tracking**: View active candidates and file sources in the sidebar.
- **Session Controls**: Clear conversations or export session transcripts.

### Option B: Interactive Terminal CLI

```bash
python cli.py
```
Interactive REPL commands:
- `exit` / `quit` — Save conversation session and exit
- `clear` — Reset conversation memory
- `help` — Display available commands

---

## 🛡️ Grounding & Anti-Hallucination Architecture

The agent enforces a **4-layer defense-in-depth** strategy against hallucinations:

1. **Instruction-Level Grounding**:
   The LLM is instructed to only answer from context retrieved via `search_cv`. This is a soft boundary — the model is guided by prompt rules, not a hard runtime enforcer.

2. **Mandatory Retrieval Gate (Graph-Level Enforcement)**:
   The LangGraph `mandatory_retrieval_node` unconditionally calls `search_cv` before the agent LLM runs. The agent cannot skip retrieval by choosing not to emit a tool call. Retrieved chunks with L2 distance score > 1.0 are flagged as `status="unavailable"` and the graph short-circuits to `fallback_node`.

3. **Graph Fallback Short-Circuit**:
   If mandatory retrieval (or any subsequent tool call) returns no relevant chunks, LangGraph bypasses the generator LLM and directly routes to `fallback_node` with:
   > *"This information is not available in the provided CV."*

4. **Fail-Closed Secondary LLM Grounding Validator**:
   Before final answer delivery, a dedicated `check_grounding_node` verifies that all claims, candidate attributions, and numeric calculations strictly match the retrieved CV excerpts. If the validator's API call fails or returns unparseable JSON, the system **fails closed** — returning `NOT_AVAILABLE_MESSAGE` rather than trusting the unverified answer.

---

## ⚙️ Configuration Reference

| Variable | Default | Description |
| :--- | :--- | :--- |
| `GOOGLE_API_KEY` | _(required)_ | Google Gemini API key |
| `LLM_PROVIDER` | `gemini` | LLM backend: `gemini`, `openai`, `ollama` |
| `LLM_MODEL` | `gemini-2.5-flash-lite` | Model identifier |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | Embedding model identifier |
| `CHUNK_SIZE` | `800` | Chunk size in characters |
| `CHUNK_OVERLAP` | `150` | Overlap between adjacent chunks |
| `RETRIEVAL_TOP_K` | `6` | Number of chunks retrieved per search |
| `RETRIEVAL_SCORE_THRESHOLD` | `1.0` | Maximum FAISS L2 distance for relevant chunks |
| `AGENT_MAX_ITERATIONS` | `5` | Maximum agent-tool execution cycles per turn |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `LOG_FILE` | `output/agent.log` | Path for structured agent activity logging |

> **Index freshness**: The FAISS index is automatically invalidated and rebuilt whenever the CV files, embedding model, chunk size, or chunk overlap change. SHA-256 hashing across all indexed files detects stale indexes — no manual deletion required.

---

## ⚠️ Assumptions & Limitations

### Assumptions
- **CV format**: CVs must be `.pdf` or `.docx`. Scanned image-only PDFs (no selectable text) will produce empty chunks and no useful answers.
- **Language**: CV content and queries are assumed to be in English.
- **API access**: A valid `GOOGLE_API_KEY` with access to both `gemini-*` (LLM) and `gemini-embedding-001` (embeddings) is required. Free-tier rate limits may cause occasional delays.
- **Single-session state**: Conversation memory is held in-process. Restarting the app or CLI clears history.
- **Local execution**: The FAISS index is stored locally in `data/faiss_index/`. There is no distributed or shared vector store.

### Limitations
- **Heuristic candidate name extraction**: The candidate's name is inferred from the first line of the document if it passes basic filters. It falls back to the filename. Complex CV layouts may mis-identify the name.
- **Best-effort section inference**: Chunk sections (e.g., `Education`, `Experience`) are assigned by keyword matching on surrounding text — not by structural PDF/DOCX parsing. Section labels may occasionally be inaccurate.
- **No authentication or multi-tenancy**: The Streamlit UI has no login. All uploaded CVs are accessible in the same session.
- **Context window limits**: Very long CVs or very large multi-CV sets may approach the LLM's context window limit during grounding validation.
- **Grounding validator is LLM-based**: The secondary validator itself calls an LLM, so it can theoretically make mistakes. The system mitigates this by failing closed on any validator error.
- **No PII protection**: CV data (names, contact details, etc.) is stored locally in plaintext in the FAISS index and log files. Do not use with sensitive personal data in production without additional safeguards.

---

## 📤 Generated Output

Every session automatically saves a structured output file to the `output/` directory.

**File naming**: `output/session_YYYYMMDD_HHMMSS.txt`

**Format**:
```
======================================================================
CV QA AGENT — SESSION OUTPUT
Session started : 2026-08-22 14:30:00
Session ended   : 2026-08-22 14:35:12
Total questions : 5
======================================================================

[2026-08-22T14:30:15] Question 1:
  Q: What is the candidate's most recent role?
  A: Based on the CV, the candidate's most recent role is Senior Software Engineer at TechCorp...

[2026-08-22T14:31:02] Question 2:
  Q: What technologies did they use there?
  A: Based on the CV, the technologies used in that role include Python, FastAPI, Kubernetes...

...
```

**CLI**: Type `exit` or `quit` to save the session automatically.
**Web UI**: Use the **💾 Export Session** button in the sidebar.

The `output/agent.log` file contains a structured, timestamped activity log including retrieval scores, grounding decisions, and latency for every turn.

---

## 🧪 Running Automated Tests

Run the complete test suite:

```bash
pytest tests/ -v
```

**Test Coverage Highlights (24 passing tests):**
- Ingestion validation (PDF/DOCX, empty documents, unsupported formats)
- Chunk metadata integrity (`candidate`, `filename`, `source`, `section`)
- Multi-turn sliding memory & transcript serialization
- Dynamic query contextualization & follow-up rewriting
- Grounding validator approval & fallback interception
- Fail-closed behaviour on empty retrieved documents
- Vector store staleness detection & hashing
