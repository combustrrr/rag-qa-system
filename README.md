# NLP Lab Experiment 5: RAG-based Question Answering System

This project is a beginner-friendly, modular implementation of a **Retrieval-Augmented Generation (RAG)** Question Answering system designed for NLP academic labs and experiments.

---

## 1. Required Libraries

All chosen libraries are **100% free and open-source**, allowing the system to run locally on CPU without mandatory paid API keys:

| Library | Primary Use in RAG Pipeline | Stage |
| :--- | :--- | :--- |
| **`pypdf`** | Lightweight, pure Python PDF reader to extract text from documents. | Stages 1 & 2 |
| **`sentence-transformers`** | Generates dense vector embeddings (e.g. `all-MiniLM-L6-v2`) locally on CPU. | Stage 5 |
| **`chromadb`** | Embedded open-source vector database for indexing and cosine similarity retrieval. | Stages 6 & 7 |
| **`transformers` & `torch`** | Local inference of instruction-tuned generative models (e.g. `google/flan-t5-base`). | Stage 9 |
| **`rouge-score` & `nltk`** | Computes evaluation metrics (Exact Match, SQuAD Token F1, ROUGE-1/2/L). | Stage 10 |
| **`rich` & `python-dotenv`** | Terminal formatting, tables, and optional environment variable management. | Utilities |

---

## 2. Installation Commands

Follow these steps to set up your environment:

### Step A: Create and Activate a Virtual Environment (Recommended)
```bash
# On Linux / macOS
python3 -m venv venv
source venv/bin/activate

# On Windows (cmd / PowerShell)
# python -m venv venv
# .\venv\Scripts\activate
```

### Step B: Install Required Packages
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step C: Verify Environment Health
Run the built-in diagnostic tool in `main.py`:
```bash
python main.py --check-env
```

---

## 3. Folder Structure

```
codespaces-blank/
│
├── data/
│   ├── sample_docs/              # Input documents to be loaded & indexed
│   │   └── attention_intro.txt   # Sample NLP text on Transformer & RAG
│   └── chromadb_storage/         # Persistent vector database store (auto-generated)
│
├── src/                          # Modular source code corresponding to each stage
│   ├── __init__.py               # Python package initialization
│   ├── document_loader.py        # [Stage 1 & 2] Document loading & text extraction (.txt, .pdf)
│   ├── text_cleaner.py           # [Stage 3] Text cleaning, sanitization & normalization
│   ├── text_chunker.py           # [Stage 4] Sliding window text chunking with overlap
│   ├── embedder.py               # [Stage 5] Dense vector embedding generation
│   ├── vector_store.py           # [Stage 6 & 7] ChromaDB indexing & similarity retrieval
│   ├── context_builder.py        # [Stage 8] Prompt template construction & context formatting
│   ├── generator.py              # [Stage 9] LLM answer generation (local / API fallback)
│   └── evaluator.py              # [Stage 10] QA metrics (Exact Match, F1, ROUGE-L)
│
├── .gitignore                    # Git ignore file for Python, caches & ChromaDB
├── requirements.txt              # Dependency specifications
├── main.py                       # CLI orchestrator & pipeline controller
└── README.md                     # Experiment guide and documentation
```

---

## 4. Explanation of What Each File Will Do

### Core Files
- **[`requirements.txt`](file:///workspaces/codespaces-blank/requirements.txt)**:
  Specifies the Python dependencies required across all 10 stages, categorized by function.
- **[`main.py`](file:///workspaces/codespaces-blank/main.py)**:
  The master entry point for the experiment. It provides CLI switches (`--check-env`, `--list-stages`, `--stage <N>`), connects all stages in sequence, and logs the execution flow.
- **[`data/sample_docs/attention_intro.txt`](file:///workspaces/codespaces-blank/data/sample_docs/attention_intro.txt)**:
  A sample knowledge-base document explaining Attention and RAG, used to test extraction, chunking, embedding, retrieval, and answering.

### Stage Modules in `src/`
- **`src/document_loader.py` (Stages 1 & 2: Document Loading & Text Extraction)**:
  Scans the `data/sample_docs/` directory, checks file types (`.txt`, `.pdf`), extracts raw string content using standard file reading or `pypdf`, and packages each file into a document object containing text and metadata (filename, page count).
- **`src/text_cleaner.py` (Stage 3: Text Cleaning)**:
  Preprocesses raw extracted text: collapses redundant whitespace, removes unwanted special characters, normalizes line breaks, and standardizes Unicode characters.
- **`src/text_chunker.py` (Stage 4: Text Chunking)**:
  Splits cleaned text into uniform chunks of size $N$ (characters or words) with a stride/overlap of $M$. Overlap ensures semantic context is preserved across chunk boundaries. Assigns a chunk ID and metadata to each piece.
- **`src/embedder.py` (Stage 5: Embedding Generation)**:
  Loads the lightweight, open-source `sentence-transformers/all-MiniLM-L6-v2` model locally on CPU. Converts textual chunks and incoming user queries into 384-dimensional dense semantic vectors.
- **`src/vector_store.py` (Stages 6 & 7: Vector Database & Similarity Retrieval)**:
  Manages a local `chromadb` client. In Stage 6, it initializes a collection and stores the chunk embeddings alongside chunk metadata and text. In Stage 7, given a query vector, it performs nearest-neighbor search (cosine similarity) to retrieve the top-$K$ most relevant chunks.
- **`src/context_builder.py` (Stage 8: Retrieved Context Construction)**:
  Assembles the prompt passed to the generator. It combines the original user question with the top-$K$ retrieved context passages and instructs the model to answer based strictly on the provided evidence.
- **`src/generator.py` (Stage 9: LLM-Based Answer Generation)**:
  Takes the synthesized prompt and produces a natural language answer. Uses an open-source, local sequence-to-sequence model (such as `google/flan-t5-base`) or an extractive fallback, eliminating the need for paid cloud APIs.
- **`src/evaluator.py` (Stage 10: Question-Answer Evaluation)**:
  Evaluates the generated response against ground-truth reference answers using standard NLP metrics:
  - **Exact Match (EM)**: Binary check of identical string match.
  - **Token-level F1**: Precision and recall over word tokens.
  - **ROUGE-L**: Longest common subsequence matching.

---

## 5. Basic Usage

```bash
# Display the 10-stage architecture
python main.py

# Check dependencies
python main.py --check-env

# List detailed breakdown of stages
python main.py --list-stages

# Run Stages 1 to 3 (Document Ingestion, Extraction & Cleaning)
python main.py --process-docs

# Run Stage 4 (Text Chunking with Overlap)
python main.py --chunk-docs --chunk-size 500 --chunk-overlap 100

# Run Stage 5 (Dense Embedding Generation)
python main.py --embed-chunks

# Or run any specific stage via --stage
python main.py --stage 5

# Run complete automated test suite
python -m unittest discover tests -v
```

---

## 6. Stage 5: Dense Vector Embedding Generation

In Stage 5, the pipeline takes the segmented text chunks produced in Stage 4 and generates dense numerical vector embeddings using the open-source **`all-MiniLM-L6-v2`** model.

### Key Features
- **Local & CPU-Friendly**: Uses `sentence-transformers` with a lightweight model (~80 MB, 384 dimensions).
- **Single-Load Architecture**: Model weights are loaded into memory once and reused across all chunks and incoming queries.
- **Traceable Mapping**: Retains complete traceability:
  $$\text{chunk\_id} \longrightarrow \text{text} \longrightarrow \text{metadata} \longrightarrow \text{embedding}$$
- **Ready for Vector Database**: Vectors are L2-normalized, enabling direct cosine similarity indexing in Stage 6 (ChromaDB).

