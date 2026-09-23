"""
Small standalone demonstration for Stage 1-3: Document Processing.
Loads documents from data/sample_docs/ and prints extracted text and metadata.
"""

import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.document_loader import DocumentLoader

def main():
    data_dir = Path(__file__).resolve().parent.parent / "data" / "sample_docs"
    print(f"Loading documents from: {data_dir}\n")

    loader = DocumentLoader(data_dir)
    documents = loader.load_directory()

    print(f"\nExtracted {len(documents)} document section(s):\n")
    for doc in documents:
        print("=" * 60)
        print(f"ID      : {doc.doc_id}")
        print(f"Source  : {doc.metadata.get('filename')} (Page {doc.metadata.get('page')}/{doc.metadata.get('total_pages')})")
        print(f"Length  : {doc.metadata.get('word_count')} words | {doc.metadata.get('char_count')} chars")
        print("-" * 60)
        print("Extracted Text:")
        print(doc.text)
        print()

if __name__ == "__main__":
    main()
