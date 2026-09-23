"""
NLP Lab Experiment 5: RAG-based Question Answering System
=========================================================
Main pipeline orchestrator and CLI entry point.

Stages Overview:
  1. Document Loading          (src/document_loader.py)
  2. Text Extraction           (src/document_loader.py)
  3. Text Cleaning             (src/text_cleaner.py)
  4. Text Chunking             (src/text_chunker.py)
  5. Embedding Generation      (src/embedder.py)
  6. Vector Database Creation  (src/vector_store.py)
  7. Similarity-based Retrieval (src/vector_store.py)
  8. Context Construction      (src/context_builder.py)
  9. LLM Answer Generation     (src/generator.py)
 10. QA Evaluation             (src/evaluator.py)
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.document_loader import DocumentLoader, Document
from src.text_cleaner import clean_text
from src.text_chunker import TextChunker, TextChunk
from src.embedder import EmbeddingGenerator, EmbeddedChunk, run_embedding_generation

DATA_DIR = BASE_DIR / "data" / "sample_docs"

# 10 Stages of Experiment 5
RAG_STAGES = [
    ("Stage 1", "Document Loading", "src/document_loader.py", "Load raw documents (.txt, .pdf) from data/sample_docs/"),
    ("Stage 2", "Text Extraction", "src/document_loader.py", "Extract raw text content and metadata (page numbers, filenames)"),
    ("Stage 3", "Text Cleaning", "src/text_cleaner.py", "Sanitize whitespace, normalize Unicode, and remove noise"),
    ("Stage 4", "Text Chunking", "src/text_chunker.py", "Split text into overlapping chunks using sliding windows"),
    ("Stage 5", "Embedding Generation", "src/embedder.py", "Generate dense vector embeddings using sentence-transformers"),
    ("Stage 6", "Vector Database Creation", "src/vector_store.py", "Index vectors and metadata into local ChromaDB collection"),
    ("Stage 7", "Similarity-based Retrieval", "src/vector_store.py", "Retrieve top-K most similar text chunks for a query"),
    ("Stage 8", "Retrieved Context Construction", "src/context_builder.py", "Format prompt template with retrieved context and query"),
    ("Stage 9", "LLM-based Answer Generation", "src/generator.py", "Generate grounded answer using open-source LLM / fallback"),
    ("Stage 10", "Question-Answer Evaluation", "src/evaluator.py", "Evaluate quality using Exact Match, Token F1, and ROUGE metrics"),
]


def print_banner():
    """Prints the project banner and the 10 stages."""
    print("=" * 80)
    print("   NLP LAB EXPERIMENT 5: RAG-BASED QUESTION ANSWERING SYSTEM")
    print("=" * 80)
    print(f" {'Stage':<10} | {'Module / File':<24} | {'Description'}")
    print("-" * 80)
    for stage_id, name, mod, desc in RAG_STAGES:
        print(f" {stage_id:<10} | {mod:<24} | {name}")
    print("=" * 80 + "\n")


def check_environment():
    """Checks whether the necessary libraries are installed in the current Python environment."""
    print("[*] Performing Dependency & Environment Health Check...\n")
    dependencies = [
        ("sentence_transformers", "sentence-transformers", "Stage 5 & 9: Vector Embeddings & HF Models"),
        ("chromadb", "chromadb", "Stage 6 & 7: Local Vector Database"),
        ("pypdf", "pypdf", "Stage 1 & 2: PDF Document Text Extraction"),
        ("rouge_score", "rouge-score", "Stage 10: ROUGE NLP Evaluation Metric"),
        ("nltk", "nltk", "Stage 10: Token F1 & String Overlap Evaluation"),
        ("rich", "rich", "CLI Formatting and Output Tables"),
    ]

    all_installed = True
    for mod_name, pkg_name, stage_desc in dependencies:
        try:
            __import__(mod_name)
            print(f"  [INSTALLED] {pkg_name:<22} ({stage_desc})")
        except ImportError:
            print(f"  [MISSING]   {pkg_name:<22} ({stage_desc})")
            all_installed = False

    print("-" * 80)
    if all_installed:
        print("[+] SUCCESS: All required libraries are installed and ready to use!\n")
    else:
        print("[!] NOTICE: Some libraries are not yet installed.")
        print("[!] Run: pip install -r requirements.txt\n")


def run_document_processing(data_dir: Path = DATA_DIR):
    """Executes Stages 1-3: File discovery, text extraction, cleaning, and metadata formatting."""
    print("=" * 80)
    print("   STAGE 1 to 3: DOCUMENT PROCESSING & EXTRACTION PIPELINE")
    print("=" * 80)
    print(f"[*] Target Directory: {data_dir}\n")

    loader = DocumentLoader(data_dir)

    # 1. File Discovery
    print("[1] Discovering Files...")
    discovered = loader.discover_files()
    print(f"    Total supported, non-empty files found: {len(discovered)}")
    for f in discovered:
        print(f"    • {f.name:<30} ({f.stat().st_size:,} bytes)")
    print()

    # 2 & 3. Extraction and Cleaning
    print("[2 & 3] Extracting & Cleaning Documents...")
    documents = loader.load_directory()
    print(f"\n[+] Successfully extracted {len(documents)} document section(s):\n")

    print("-" * 80)
    for idx, doc in enumerate(documents, start=1):
        m = doc.metadata
        print(f"Document #{idx}:")
        print(f"  • Doc ID     : {doc.doc_id}")
        print(f"  • Source     : {m.get('filename')}")
        print(f"  • Type       : {m.get('file_type', '').upper()}")
        print(f"  • Page       : {m.get('page')} of {m.get('total_pages')}")
        print(f"  • Word Count : {m.get('word_count')} words ({m.get('char_count')} chars)")
        print("  • Cleaned Text Preview:")
        preview_lines = doc.text.split("\n")[:3]
        for line in preview_lines:
            print(f"    | {line[:72]}")
        if len(doc.text.split("\n")) > 3:
            print("    | ...")
        print("-" * 80)

    print(f"\n[✓] Document Processing complete! Extracted {len(documents)} units ready for chunking (Stage 4).\n")
    return documents


def print_chunking_theory():
    """Prints a brief explanation of why chunking and overlap are required in RAG."""
    print("=" * 80)
    print("   THEORETICAL INSIGHT: WHY CHUNKING & OVERLAP ARE REQUIRED IN RAG")
    print("=" * 80)
    print("1. Why Chunking is Required:")
    print("   • Embedding Constraint : Embedding models (e.g. MiniLM) have fixed token limits")
    print("                            (typically 256 or 512 tokens). Full documents get truncated.")
    print("   • Retrieval Precision  : User queries target specific facts, not entire books.")
    print("                            Smaller chunks allow pinpoint nearest-neighbor search.")
    print("   • Context Window Limits: Feeding entire documents to an LLM wastes tokens, adds")
    print("                            latency, and increases hallucination risk.")
    print("\n2. Why Overlap is Required:")
    print("   • Context Continuity   : Key ideas, definitions, and clauses often span across cut points.")
    print("                            Overlap ensures boundary information is not split in half.")
    print("   • High Retrieval Recall: Queries targeting transition sentences can still match")
    print("                            adjacent chunks with complete semantic meaning.")
    print("=" * 80 + "\n")


def run_text_chunking(data_dir: Path = DATA_DIR, chunk_size: int = 500, chunk_overlap: int = 100):
    """Executes Stage 4: Text Chunking on extracted documents."""
    # First extract documents
    loader = DocumentLoader(data_dir)
    documents = loader.load_directory()

    if not documents:
        print("[!] No documents found to chunk. Run document processing first.")
        return []

    print("=" * 80)
    print("   STAGE 4: TEXT CHUNKING PIPELINE")
    print("=" * 80)
    print(f"[*] Configuration: chunk_size = {chunk_size} chars | chunk_overlap = {chunk_overlap} chars (~{int(chunk_overlap/chunk_size*100)}% overlap)\n")

    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks = chunker.chunk_documents(documents)

    # 1. Total documents and chunks
    print(f"[*] Total Documents Processed : {len(documents)}")
    print(f"[*] Total Text Chunks Created : {len(all_chunks)}\n")

    # 2. Display first 3 chunks and their complete metadata
    display_count = min(3, len(all_chunks))
    print(f"[+] Displaying First {display_count} Chunks with Metadata:")
    print("-" * 80)

    for i in range(display_count):
        chunk = all_chunks[i]
        meta = chunk.metadata
        print(f"Chunk #{i + 1}:")
        print(f"  • Chunk ID        : {chunk.chunk_id}")
        print(f"  • Parent Doc ID   : {chunk.doc_id}")
        print(f"  • Source Filename : {meta.get('filename')}")
        print(f"  • Page Number     : {meta.get('page')} of {meta.get('total_pages')}")
        print(f"  • File Format     : {meta.get('file_type', '').upper()}")
        print(f"  • Chunk Index     : {chunk.chunk_index + 1} of {meta.get('total_chunks_in_doc')}")
        print(f"  • Character Range : [{meta.get('start_char')} : {meta.get('end_char')}] ({meta.get('char_count')} chars, {meta.get('word_count')} words)")
        print("  • Chunk Text Content:")
        for line in chunk.text.split("\n"):
            print(f"    | {line}")
        print("-" * 80)

    print()
    print_chunking_theory()
    print(f"[✓] Stage 4 complete! {len(all_chunks)} chunks ready for Embedding Generation (Stage 5).\n")
    return all_chunks


def run_stage_stub(stage_num: int, chunk_size: int = 500, chunk_overlap: int = 100, model_name: str = "all-MiniLM-L6-v2"):
    """Runs implemented stages or explains pending ones."""
    if stage_num in (1, 2, 3):
        run_document_processing()
        return

    if stage_num == 4:
        run_text_chunking(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return

    if stage_num == 5:
        run_embedding_generation(chunk_size=chunk_size, chunk_overlap=chunk_overlap, model_name=model_name)
        return

    stage_id, name, module_path, desc = RAG_STAGES[stage_num - 1]
    print(f"[*] Selected: {stage_id} - {name}")
    print(f"    Target Module : {module_path}")
    print(f"    Objective     : {desc}")
    print("    Status        : Scheduled for subsequent implementation.")
    print("                    Stages 1 to 5 (Document Processing, Chunking & Embeddings) are currently implemented.\n")


def main():
    parser = argparse.ArgumentParser(
        description="NLP Lab Experiment 5: RAG Question Answering System Orchestrator"
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Check Python environment and package dependencies",
    )
    parser.add_argument(
        "--list-stages",
        action="store_true",
        help="List all 10 stages of the RAG experiment",
    )
    parser.add_argument(
        "--process-docs",
        action="store_true",
        help="Run Stages 1-3: Load, extract, and clean documents from data/sample_docs/",
    )
    parser.add_argument(
        "--chunk-docs",
        action="store_true",
        help="Run Stage 4: Split documents into overlapping chunks with metadata",
    )
    parser.add_argument(
        "--embed-chunks",
        action="store_true",
        help="Run Stage 5: Generate dense vector embeddings using sentence-transformers",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence-transformers model name (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Maximum characters per chunk (default: 500)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Overlap characters between adjacent chunks (default: 100)",
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=range(1, 11),
        help="Inspect or run a specific stage (1 to 10)",
    )
    args = parser.parse_args()

    print_banner()

    if args.check_env:
        check_environment()
        return

    if args.list_stages:
        print("Detailed 10 Stages Breakdown:")
        for stage_id, name, mod, desc in RAG_STAGES:
            print(f"  [{stage_id}] {name}")
            print(f"      File: {mod}")
            print(f"      Goal: {desc}\n")
        return

    if args.process_docs:
        run_document_processing()
        return

    if args.chunk_docs:
        run_text_chunking(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
        return

    if args.embed_chunks:
        run_embedding_generation(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            model_name=args.embedding_model,
        )
        return

    if args.stage:
        run_stage_stub(
            args.stage,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            model_name=args.embedding_model,
        )
        return

    # Default view if no arguments provided
    print("[i] Project setup initialized successfully!")
    print("[i] Available commands:")
    print("    python main.py --embed-chunks   Run Stage 5 (Embedding Generation)")
    print("    python main.py --chunk-docs     Run Stage 4 (Text Chunking)")
    print("    python main.py --process-docs   Run Stages 1-3 (Document Processing)")
    print("    python main.py --check-env      Check required library installations")
    print("    python main.py --list-stages    View the 10 RAG stages and mapped files")
    print("    python main.py --stage <1-10>   Run or inspect a specific RAG stage")
    print("    python -m unittest              Run automated test suite\n")


if __name__ == "__main__":
    main()
