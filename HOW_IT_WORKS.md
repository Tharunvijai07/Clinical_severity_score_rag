# MedSeverity AI – How It Works

## 🏥 Overview

**MedSeverity AI** is a **Retrieval-Augmented Generation (RAG)** system that analyzes patient clinical data (lab reports or text) and generates a structured severity assessment grounded in medical guidelines. It combines:

- **Vector embeddings** (ChromaDB + sentence-transformers) to retrieve relevant medical guidelines
- **OpenRouter-hosted LLM** to reason over patient findings + retrieved evidence
- **Streamlit UI** for interactive analysis

---

## 🎯 Core Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│ USER INPUT (Step 1)                                              │
│  • Upload PDF lab report  OR  Enter clinical data as text       │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ EXTRACTION (Step 2)                                              │
│  • PDF → Text (PyMuPDF) or Text → Stored as-is                 │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ KNOWLEDGE BASE BUILD (One-time setup)                           │
│  • Scan knowledge_base/ for .txt/.pdf medical guidelines        │
│  • Split each doc into chunks (600 tokens, 80-token overlap)   │
│  • Embed chunks using sentence-transformers                    │
│  • Store embeddings + metadata in ChromaDB                     │
└─────────────────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ RETRIEVAL (Step 3)                                               │
│  • Embed patient report text using same encoder                │
│  • Query ChromaDB for top-K similar chunks                     │
│  • Return ranked list of relevant medical guidelines           │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ ANALYSIS (Step 4)                                                │
│  • Build LLM prompt: patient text + retrieved guidelines        │
│  • Call OpenRouter model for analysis                           │
│  • Parse JSON response:                                         │
│    - severity_score (0–10)                                     │
│    - severity_level (Low/Moderate/High/Critical)              │
│    - key_findings (list of abnormalities)                     │
│    - evidence (citations from retrieved guidelines)           │
│    - summary (clinical assessment)                             │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ DISPLAY (Streamlit UI)                                           │
│  • Severity gauge with color-coded level                       │
│  • Key findings & medical evidence                             │
│  • Clinical summary & references                               │
│  • JSON download option                                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Architecture & Components

### **1. RAG Pipeline** (`rag/rag_pipeline.py`)
**Purpose:** Orchestrates knowledge base setup and per-request retrieval.

**Key Methods:**
- `build_knowledge_base()` — Scans `knowledge_base/`, chunks documents, embeds, stores in ChromaDB
- `run_query(patient_text, top_k)` — Retrieves top-K relevant medical guideline chunks
- `is_knowledge_base_ready()` — Checks if vector store has documents

**Inputs:** Patient clinical text (from Step 2)  
**Outputs:** List of `RetrievedChunk` objects (text, source, similarity score)

---

### **2. Vector Store** (`rag/vector_store.py`)
**Purpose:** Manages ChromaDB (persistent vector database).

**Key Methods:**
- `add_documents()` — Upsert text chunks + embeddings into ChromaDB
- `query()` — Semantic search by embedding similarity
- `reset()` — Clear the collection for rebuilds
- `count()` — Total chunks stored

**Storage Location:** `./vectordb/` (ChromaDB persistent directory)

---

### **3. Retriever** (`rag/retriever.py`)
**Purpose:** High-level query interface with deduplication.

**Process:**
1. Embeds query text using sentence-transformers (`all-MiniLM-L6-v2`)
2. Queries VectorStore for top-K most similar chunks
3. Deduplicates by content hash (same passage ≠ duplicate results)
4. Returns ranked `RetrievedChunk` objects with similarity scores

**Output:** `List[RetrievedChunk]` sorted by relevance

---

### **4. Text Processing**
- **PDF Parser** (`rag/pdf_parser.py`) — Extracts text from PDFs using PyMuPDF
- **Text Chunker** (`rag/text_chunker.py`) — Splits long documents into semantic chunks (600 tokens, 80-token overlap)

---

### **5. LLM Integration** (`llm/`)

#### **OpenRouter Client** (`llm/gemini_client.py`)
- Calls OpenRouter's OpenAI-compatible chat completions API
- Handles authentication via `OPENROUTER_API_KEY` (from `.env`)
- Returns raw text response from model

#### **Severity Analyzer** (`llm/severity_analyzer.py`)
**Purpose:** Orchestrates the full LLM analysis pipeline.

**Process:**
1. Builds a structured prompt combining:
   - System prompt (role definition)
   - Patient clinical data
   - Retrieved medical guidelines
2. Calls the configured OpenRouter model
3. Parses JSON response into `SeverityResult` dataclass
4. Validates all fields (score 0–10, level in {Low, Moderate, High, Critical})

**Output:** 
```python
@dataclass
class SeverityResult:
    severity_score: int          # 0–10
    severity_level: str          # Low | Moderate | High | Critical
    key_findings: List[str]      # Abnormalities detected
    evidence: List[str]          # Citations from guidelines
    summary: str                 # Clinical assessment
    raw_response: str            # Raw model output
    parse_error: str             # Error message if parsing failed
```

---

### **6. Prompts** (`llm/prompts.py`)
Defines:
- **System prompt** — Instructs the LLM to act as a clinical reasoning engine
- **Severity prompt builder** — Combines patient text + retrieved guidelines into a structured prompt

**Example prompt structure:**
```
[SYSTEM]
You are an expert clinical analyzer. Based on patient findings and 
medical guidelines, assess clinical severity (0–10 scale).

[PATIENT DATA]
<extracted clinical text>

[RETRIEVED MEDICAL GUIDELINES]
1. <guideline source>: <relevant passage>
2. <guideline source>: <relevant passage>
...

[TASK]
Provide a JSON response with:
  - severity_score: int (0-10)
  - severity_level: str (Low | Moderate | High | Critical)
  - key_findings: List[str]
  - evidence: List[str] (cite which guideline each came from)
  - summary: str
```

---

### **7. Configuration** (`utils/config.py`)
Loads environment variables from `.env`:
- `OPENROUTER_API_KEY` — OpenRouter API authentication
- `OPENROUTER_MODEL` — Model slug (default: `openai/gpt-4o-mini`)
- `OPENROUTER_BASE_URL` — OpenRouter API base URL (default: `https://openrouter.ai/api/v1`)
- `CHROMA_DB_PATH` — Vector store directory
- `KNOWLEDGE_BASE_PATH` — Medical guidelines directory
- `EMBEDDING_MODEL` — Sentence encoder (default: `all-MiniLM-L6-v2`)
- `CHUNK_SIZE` — Text chunk size in tokens (default: 600)
- `CHUNK_OVERLAP` — Overlap between chunks (default: 80)
- `TOP_K_RESULTS` — Default retrieval count (default: 5)

---

### **8. Streamlit App** (`app/streamlit_app.py`)
**Purpose:** Interactive web UI.

**Features:**
- **Step 1:** Dual input (PDF upload + text entry)
- **Step 2:** Document extraction & metrics
- **Step 3:** Knowledge base retrieval with relevance scores
- **Step 4:** Severity assessment with LLM analysis
- **Display:** Color-coded severity gauge, findings, evidence, references
- **Export:** JSON download of results

---

## 📊 Data Flow Diagram

```
┌─────────────────────────┐
│ Medical Guidelines      │
│ (5 .txt files)         │
└────────┬────────────────┘
         │ build_knowledge_base()
         ▼
┌─────────────────────────┐
│ Text Chunker            │
│ (600 tok, 80 overlap)  │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Embedding Model         │
│ (all-MiniLM-L6-v2)     │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ ChromaDB Vector Store   │
│ (persistent)            │
└─────────────────────────┘
         ▲
         │ query() with embedding
         │
┌────────┴────────────────┐
│ Patient Data (Step 2)   │
│ (PDF or Text)          │
└────────┬────────────────┘
         │ embed patient text
         ▼
┌─────────────────────────┐
│ Retriever               │
│ (top-K similar chunks)  │
└────────┬────────────────┘
         │ retrieved_chunks
         ▼
┌─────────────────────────┐
│ Build LLM Prompt        │
│ (patient + guidelines)  │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ OpenRouter Model        │
│ (LLM reasoning)         │
└────────┬────────────────┘
         │ raw response
         ▼
┌─────────────────────────┐
│ Parse JSON              │
│ (SeverityResult)        │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Streamlit UI            │
│ (display & export)      │
└─────────────────────────┘
```

---

## 📚 Knowledge Base

**Location:** `./knowledge_base/`

**Current Documents:**
- `critical_care_protocols.txt` — ICU & critical illness guidelines
- `emergency_medicine_references.txt` — Emergency department protocols
- `kidney_disease_guidelines.txt` — Renal failure & hyperkalemia protocols
- `liver_failure_guidelines.txt` — Hepatic dysfunction guidelines
- `sepsis_guidelines.txt` — Sepsis diagnosis & management

**One-time Setup:**
The app's **Build Knowledge Base** button (or `setup_knowledge_base.py` script) will:
1. Read all `.txt` and `.pdf` files from `knowledge_base/`
2. Split into overlapping chunks
3. Embed using `all-MiniLM-L6-v2`
4. Store in ChromaDB at `./vectordb/`

After build, the vector store is persistent — no rebuild needed unless `--force` flag is used.

---

## 🔑 Environment Setup

**`.env` file required:**
```bash
OPENROUTER_API_KEY=<your-openrouter-api-key>
OPENROUTER_MODEL=openai/gpt-4o-mini
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
CHROMA_DB_PATH=./vectordb
KNOWLEDGE_BASE_PATH=./knowledge_base
UPLOADS_PATH=./uploads
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=600
CHUNK_OVERLAP=80
TOP_K_RESULTS=5
```

---

## 🚀 Running the Project

### **Step 1: Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Step 2: Set API Key**
Update `.env` with your OpenRouter API key from [openrouter.ai](https://openrouter.ai/keys)

### **Step 3: Start Streamlit**
```bash
streamlit run app/streamlit_app.py
```

### **Step 4: Build Knowledge Base (first time only)**
In the Streamlit sidebar:
- Click **🔨 Build Knowledge Base**
- Wait for ingestion to complete

### **Step 5: Use the App**
1. **Step 1:** Enter patient data (text or PDF)
2. **Step 2:** Auto-extracted or ready for review
3. **Step 3:** Click "🔍 Retrieve Medical Guidelines"
4. **Step 4:** Click "🧠 Generate Clinical Severity Assessment"
5. View results, download JSON, review references

---

## 🔑 Key Concepts

### **RAG (Retrieval-Augmented Generation)**
Instead of:
- ❌ Asking LLM to reason from training data alone (outdated, hallucinations)

We:
- ✅ Retrieve relevant expert guidelines from a vector store
- ✅ Give LLM the guidelines + patient data
- ✅ LLM reasons over ground-truth medical knowledge
- ✅ Result is evidence-grounded severity assessment

### **Embeddings**
- **Sentence-transformers** (`all-MiniLM-L6-v2`) converts text → 384-dimensional vectors
- Semantic similarity = cosine distance between vectors
- Used for both guidelines (knowledge base) and queries (patient text)

### **ChromaDB**
- Lightweight, persistent vector database
- Stores text chunks + embeddings + metadata
- Query returns ranked results by cosine similarity
- No external database needed; stores locally in `./vectordb/`

### **Chunking Strategy**
- **Size:** 600 tokens (~400–500 words)
- **Overlap:** 80 tokens (ensures context continuity)
- **Effect:** Long documents split into semantic passages that preserve context across chunk boundaries

---

## 📁 File Structure

```
clinical-severity-rag/
├── app/
│   ├── __init__.py
│   └── streamlit_app.py          # Main UI
├── rag/
│   ├── __init__.py
│   ├── rag_pipeline.py           # Orchestrator
│   ├── vector_store.py           # ChromaDB interface
│   ├── retriever.py              # Query interface
│   ├── pdf_parser.py             # PDF → text
│   ├── text_chunker.py           # Document chunking
│   └── embedder.py               # Embedding wrapper
├── llm/
│   ├── __init__.py
│   ├── gemini_client.py          # OpenRouter API wrapper
│   ├── severity_analyzer.py      # LLM orchestration
│   └── prompts.py                # Prompt templates
├── utils/
│   ├── __init__.py
│   ├── config.py                 # Environment & config
│   └── logger.py                 # Logging setup
├── knowledge_base/               # Medical guideline documents
│   ├── critical_care_protocols.txt
│   ├── emergency_medicine_references.txt
│   ├── kidney_disease_guidelines.txt
│   ├── liver_failure_guidelines.txt
│   └── sepsis_guidelines.txt
├── vectordb/                     # ChromaDB storage (auto-created)
├── uploads/                      # PDF uploads storage
├── .env                          # Environment variables (you create this)
├── .env.example                  # Template for .env
├── requirements.txt              # Python dependencies
├── setup_knowledge_base.py       # CLI to build KB (alternative to Streamlit UI)
└── README.md                     # Project overview
```

---

## 🔄 Typical User Journey

1. **Doctor/Clinician opens app** → Sees 4 steps
2. **Step 1:** Pastes a patient case (or uploads lab PDF)
3. **Step 2:** Text is auto-extracted or entered
4. **Step 3:** Clicks "Retrieve Medical Guidelines" → App queries vector store → Shows top-5 relevant guidelines with similarity scores
5. **Step 4:** Clicks "Generate Assessment" → OpenRouter analyzes patient data + guidelines → Returns:
   - **Severity Score** (0–10 with color gauge)
   - **Severity Level** (Low/Moderate/High/Critical)
   - **Key Findings** (abnormalities detected in patient data)
   - **Medical Evidence** (citing which guideline supports each finding)
   - **Clinical Summary** (reasoned assessment)
6. **Export & Review:**
   - View retrieved guidelines in detail
   - Download assessment as JSON
   - Share with team

---

## 🛠️ Customization

### **Add More Medical Guidelines**
1. Add `.txt` or `.pdf` files to `./knowledge_base/`
2. Click "🔨 Build Knowledge Base" with **Force rebuild** checked
3. New documents will be chunked, embedded, and added to ChromaDB

### **Adjust Retrieval Sensitivity**
In Streamlit sidebar **Settings** → **Retrieved chunks (top-K)**
- Lower (3) = more focused results
- Higher (10) = broader coverage

### **Change Model**
Edit `.env`:
```bash
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet  # Or another OpenRouter model slug
```

### **Change Embedding Model**
Edit `.env`:
```bash
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
```
Then rebuild knowledge base.

---

## 🐛 Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| "Knowledge base not ready" | KB hasn't been built | Click "Build Knowledge Base" in sidebar |
| "Invalid API key" | Missing or wrong OPENROUTER_API_KEY | Update `.env` with your key from openrouter.ai |
| Slow retrieval | Large KB, slow embedding | Adjust CHUNK_SIZE in `.env` or use GPU if available |
| Poor severity predictions | Irrelevant guidelines | Add more domain-specific documents to `knowledge_base/` |
| Memory issues | Large PDFs or KB | Increase CHUNK_SIZE or reduce TOP_K_RESULTS |

---

## 📖 Summary

**MedSeverity AI** is a production-ready **clinical decision support system** that:
1. ✅ Ingests medical guidelines (one-time setup)
2. ✅ Retrieves relevant evidence for each patient case
3. ✅ Uses LLM reasoning to assess severity with medical grounding
4. ✅ Provides explainable results (shows which guidelines were used)
5. ✅ Exports structured JSON for downstream integration

The system is **evidence-grounded** (not hallucinating), **traceable** (shows sources), and **modular** (easy to add guidelines, swap models, adjust thresholds).
