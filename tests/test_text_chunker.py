"""
Unit tests for Stage 4: Text Chunking.
"""

import unittest
from src.document_loader import Document
from src.text_chunker import TextChunker, TextChunk


class TestTextChunker(unittest.TestCase):
    def setUp(self):
        self.chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        self.sample_text = (
            "Natural Language Processing (NLP) is a branch of artificial intelligence that helps computers "
            "understand, interpret, and manipulate human language. Retrieval-Augmented Generation (RAG) is "
            "an advanced technique that combines the powers of pre-trained dense language models with external "
            "knowledge retrieval mechanisms. By fetching relevant passages from a vector database, RAG grounds "
            "the generated answers in verified facts, drastically reducing hallucinations."
        )
        self.sample_doc = Document(
            doc_id="test_doc_001",
            text=self.sample_text,
            metadata={
                "filename": "nlp_overview.txt",
                "page": 1,
                "total_pages": 1,
                "file_type": "txt",
            },
        )

    def test_parameter_validation(self):
        """Verify invalid chunk_size and chunk_overlap raise ValueErrors."""
        with self.assertRaises(ValueError):
            TextChunker(chunk_size=-10, chunk_overlap=5)

        with self.assertRaises(ValueError):
            TextChunker(chunk_size=100, chunk_overlap=100)

        with self.assertRaises(ValueError):
            TextChunker(chunk_size=100, chunk_overlap=150)

    def test_empty_and_whitespace_prevention(self):
        """Ensure no empty chunks are generated from empty or blank text."""
        empty_doc = Document(doc_id="empty", text="", metadata={})
        blank_doc = Document(doc_id="blank", text="    \n\n\t   ", metadata={})

        self.assertEqual(len(self.chunker.chunk_document(empty_doc)), 0)
        self.assertEqual(len(self.chunker.chunk_document(blank_doc)), 0)

    def test_short_document_single_chunk(self):
        """A document shorter than chunk_size should produce exactly one chunk."""
        short_doc = Document(
            doc_id="short_01",
            text="Short text.",
            metadata={"filename": "short.txt", "page": 1},
        )
        chunks = self.chunker.chunk_document(short_doc)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "Short text.")
        self.assertEqual(chunks[0].chunk_id, "short_01_c000")
        self.assertEqual(chunks[0].chunk_index, 0)

    def test_metadata_preservation(self):
        """Ensure parent metadata is retained and chunk-specific metadata is attached."""
        chunks = self.chunker.chunk_document(self.sample_doc)
        self.assertGreater(len(chunks), 1)

        for chunk in chunks:
            self.assertIsInstance(chunk, TextChunk)
            self.assertEqual(chunk.doc_id, "test_doc_001")
            self.assertEqual(chunk.metadata["filename"], "nlp_overview.txt")
            self.assertEqual(chunk.metadata["page"], 1)
            self.assertEqual(chunk.metadata["file_type"], "txt")
            self.assertEqual(chunk.metadata["parent_doc_id"], "test_doc_001")
            self.assertIn("start_char", chunk.metadata)
            self.assertIn("end_char", chunk.metadata)
            self.assertIn("word_count", chunk.metadata)
            self.assertIn("char_count", chunk.metadata)
            self.assertGreater(chunk.metadata["char_count"], 0)

    def test_unique_chunk_ids(self):
        """Ensure all generated chunk IDs are strictly unique."""
        chunks = self.chunker.chunk_document(self.sample_doc)
        ids = [c.chunk_id for c in chunks]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(c.chunk_id.startswith("test_doc_001_c") for c in chunks))

    def test_chunk_overlap_presence(self):
        """Check that adjacent chunks share text corresponding to the overlap."""
        chunks = self.chunker.chunk_document(self.sample_doc)
        for i in range(len(chunks) - 1):
            curr_text = chunks[i].text
            next_text = chunks[i + 1].text
            # Check overlap tokens
            curr_words = curr_text.split()
            next_words = next_text.split()
            # There should be shared words between the end of chunk i and start of chunk i+1
            shared = set(curr_words[-5:]) & set(next_words[:5])
            self.assertTrue(
                len(shared) > 0,
                f"Expected shared words between Chunk {i} and {i+1}, got none.",
            )

    def test_chunk_documents_batch(self):
        """Test batch chunking on multiple documents."""
        doc2 = Document(
            doc_id="test_doc_002",
            text="Second document text with some more information to chunk.",
            metadata={"filename": "doc2.txt", "page": 1},
        )
        all_chunks = self.chunker.chunk_documents([self.sample_doc, doc2])
        self.assertGreater(len(all_chunks), len(self.chunker.chunk_document(self.sample_doc)))


if __name__ == "__main__":
    unittest.main()
