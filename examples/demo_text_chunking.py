"""
Standalone demonstration for Stage 4: Text Chunking.
Loads documents, chunks them with sliding windows, and prints chunk details.
"""

import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.document_loader import DocumentLoader
from src.text_chunker import TextChunker


def main():
    data_dir = BASE_DIR / "data" / "sample_docs"
    print(f"Loading documents from: {data_dir}\n")

    loader = DocumentLoader(data_dir)
    documents = loader.load_directory()

    chunker = TextChunker(chunk_size=500, chunk_overlap=100)
    chunks = chunker.chunk_documents(documents)

    print(f"Total Documents : {len(documents)}")
    print(f"Total Chunks    : {len(chunks)}\n")

    display_count = min(3, len(chunks))
    print(f"Displaying First {display_count} Chunks:\n")

    for i in range(display_count):
        c = chunks[i]
        m = c.metadata
        print("=" * 70)
        print(f"Chunk #{i + 1} ID   : {c.chunk_id}")
        print(f"Parent Doc    : {c.doc_id}")
        print(f"Source File   : {m.get('filename')} (Page {m.get('page')}/{m.get('total_pages')})")
        print(f"Position      : Chunk {c.chunk_index + 1} of {m.get('total_chunks_in_doc')}")
        print(f"Range         : [{m.get('start_char')} : {m.get('end_char')}] ({m.get('char_count')} chars, {m.get('word_count')} words)")
        print("-" * 70)
        print("Text:")
        print(c.text)
        print()


if __name__ == "__main__":
    main()
