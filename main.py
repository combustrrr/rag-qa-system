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

# Directory references
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "sample_docs"
CHROMA_DIR = BASE_DIR / "data" / "chromadb_storage"
SRC_DIR = BASE_DIR / "src"

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
    print("=" * 78)
    print("   NLP LAB EXPERIMENT 5: RAG-BASED QUESTION ANSWERING SYSTEM")
    print("=" * 78)
    print(f" {'Stage':<10} | {'Module / File':<24} | {'Description'}")
    print("-" * 78)
    for stage_id, name, mod, desc in RAG_STAGES:
        print(f" {stage_id:<10} | {mod:<24} | {name}")
    print("=" * 78 + "\n")


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

    print("-" * 78)
    if all_installed:
        print("[+] SUCCESS: All required libraries are installed and ready to use!\n")
    else:
        print("[!] NOTICE: Some libraries are not yet installed.")
        print("[!] Run: pip install -r requirements.txt\n")


def run_stage_stub(stage_num: int):
    """Placeholder runner for individual stages."""
    stage_id, name, module_path, desc = RAG_STAGES[stage_num - 1]
    print(f"[*] Selected: {stage_id} - {name}")
    print(f"    Target Module : {module_path}")
    print(f"    Objective     : {desc}")
    print("    Status        : Initial project scaffolding complete.")
    print("                    Implementation of this stage will be added in subsequent steps.\n")


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

    if args.stage:
        run_stage_stub(args.stage)
        return

    # Default view if no arguments provided
    print("[i] Project setup initialized successfully!")
    print("[i] Available commands:")
    print("    python main.py --check-env      Check required library installations")
    print("    python main.py --list-stages    View the 10 RAG stages and mapped files")
    print("    python main.py --stage <1-10>   Inspect a specific RAG stage")
    print("\n[i] Ready to proceed with Stage 1 & 2 implementation when requested.\n")


if __name__ == "__main__":
    main()
