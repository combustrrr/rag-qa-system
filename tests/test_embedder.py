"""
NLP Lab Experiment 5: Stage 5 - Embedding Generation Unit & Integration Tests
=============================================================================
Tests the functionality of src/embedder.py:
- EmbeddingGenerator initialization and single-load mechanism
- Dense vector generation and expected dimensions (384 for all-MiniLM-L6-v2)
- Maintenance of chunk_id -> text -> metadata -> embedding mapping
- EmbeddedChunk data structures and serialization
- Handling of edge cases (empty strings, empty chunk lists)
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from src.text_chunker import TextChunk
from src.embedder import EmbeddedChunk, EmbeddingGenerator, DEFAULT_MODEL_NAME


class TestEmbeddedChunkDataclass(unittest.TestCase):
    """Tests for the EmbeddedChunk data structure."""

    def setUp(self):
        self.sample_embedding = [0.12, -0.45, 0.78, -0.01]
        self.chunk = EmbeddedChunk(
            chunk_id="doc001_c000",
            text="Natural language processing powers modern RAG systems.",
            metadata={"filename": "sample.txt", "page": 1, "char_count": 55},
            embedding=self.sample_embedding,
        )

    def test_attributes_and_dimension(self):
        """Verifies basic fields and dimension computation."""
        self.assertEqual(self.chunk.chunk_id, "doc001_c000")
        self.assertEqual(self.chunk.text, "Natural language processing powers modern RAG systems.")
        self.assertEqual(self.chunk.embedding_dim, 4)
        self.assertEqual(self.chunk.metadata["filename"], "sample.txt")
        self.assertEqual(self.chunk.embedding, self.sample_embedding)

    def test_to_dict_structure(self):
        """Verifies dictionary serialization structure."""
        data = self.chunk.to_dict()
        self.assertIn("chunk_id", data)
        self.assertIn("text", data)
        self.assertIn("metadata", data)
        self.assertIn("embedding", data)
        self.assertEqual(data["chunk_id"], "doc001_c000")
        self.assertEqual(data["embedding"], self.sample_embedding)


class TestEmbeddingGeneratorMocked(unittest.TestCase):
    """Fast unit tests using mocked SentenceTransformer to isolate logic."""

    @patch("sentence_transformers.SentenceTransformer")
    def test_single_model_load_and_caching(self, mock_st_cls):
        """Ensures the underlying transformer is loaded exactly once upon initialization."""
        mock_instance = MagicMock()
        mock_instance.get_embedding_dimension.return_value = 384
        mock_instance.get_sentence_embedding_dimension.return_value = 384
        mock_st_cls.return_value = mock_instance

        generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")

        # SentenceTransformer must have been called exactly once
        mock_st_cls.assert_called_once_with("all-MiniLM-L6-v2", device=None)
        self.assertEqual(generator.embedding_dimension, 384)
        self.assertEqual(generator.model_name, "all-MiniLM-L6-v2")

    @patch("sentence_transformers.SentenceTransformer")
    def test_embed_text_single(self, mock_st_cls):
        """Tests single string encoding."""
        mock_instance = MagicMock()
        mock_instance.get_embedding_dimension.return_value = 384
        mock_instance.get_sentence_embedding_dimension.return_value = 384
        dummy_vec = np.zeros(384, dtype=np.float32)
        mock_instance.encode.return_value = dummy_vec
        mock_st_cls.return_value = mock_instance

        generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
        result = generator.embed_text("Sample query")

        self.assertEqual(len(result), 384)
        mock_instance.encode.assert_called_once()

    @patch("sentence_transformers.SentenceTransformer")
    def test_embed_chunks_and_mapping(self, mock_st_cls):
        """Tests converting TextChunks into EmbeddedChunks and verifying the full mapping."""
        mock_instance = MagicMock()
        mock_instance.get_embedding_dimension.return_value = 4
        mock_instance.get_sentence_embedding_dimension.return_value = 4
        # Return 2 mock 4D vectors
        mock_instance.encode.return_value = np.array([
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
        ], dtype=np.float32)
        mock_st_cls.return_value = mock_instance

        generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")

        chunks = [
            TextChunk(
                chunk_id="chunk_1",
                text="First chunk content",
                doc_id="doc1",
                chunk_index=0,
                metadata={"filename": "doc1.txt", "page": 1},
            ),
            TextChunk(
                chunk_id="chunk_2",
                text="Second chunk content",
                doc_id="doc1",
                chunk_index=1,
                metadata={"filename": "doc1.txt", "page": 1},
            ),
        ]

        embedded = generator.embed_chunks(chunks)
        self.assertEqual(len(embedded), 2)
        self.assertEqual(embedded[0].chunk_id, "chunk_1")
        np.testing.assert_allclose(embedded[0].embedding, [0.1, 0.2, 0.3, 0.4], atol=1e-5)
        self.assertEqual(embedded[0].metadata["embedding_model"], "all-MiniLM-L6-v2")
        self.assertEqual(embedded[0].metadata["embedding_dim"], 4)

        # Verify mapping chunk_id -> text -> metadata -> embedding
        mapping = generator.get_mapping(embedded)
        self.assertIn("chunk_1", mapping)
        self.assertIn("chunk_2", mapping)
        self.assertEqual(mapping["chunk_1"]["text"], "First chunk content")
        self.assertEqual(mapping["chunk_1"]["metadata"]["filename"], "doc1.txt")
        np.testing.assert_allclose(mapping["chunk_1"]["embedding"], [0.1, 0.2, 0.3, 0.4], atol=1e-5)

    @patch("sentence_transformers.SentenceTransformer")
    def test_empty_inputs(self, mock_st_cls):
        """Tests that empty chunk lists return empty results without error."""
        mock_instance = MagicMock()
        mock_instance.get_embedding_dimension.return_value = 384
        mock_instance.get_sentence_embedding_dimension.return_value = 384
        mock_st_cls.return_value = mock_instance

        generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
        self.assertEqual(generator.embed_chunks([]), [])
        self.assertEqual(generator.embed_texts([]), [])
        self.assertEqual(generator.get_mapping([]), {})


class TestEmbeddingGeneratorLiveIntegration(unittest.TestCase):
    """Integration test with actual all-MiniLM-L6-v2 model."""

    @classmethod
    def setUpClass(cls):
        """Attempt to load the real model once for integration testing."""
        try:
            cls.generator = EmbeddingGenerator(model_name=DEFAULT_MODEL_NAME)
            cls.live_available = True
        except Exception:
            cls.live_available = False

    def test_live_embedding_generation(self):
        """Tests live model generation if dependencies and model weights are accessible."""
        if not self.live_available:
            self.skipTest("Live SentenceTransformer model could not be loaded in this test environment.")

        # Test single text embedding
        text = "Retrieval-Augmented Generation bridges pre-trained models with external facts."
        emb = self.generator.embed_text(text)
        self.assertEqual(len(emb), 384)
        self.assertTrue(all(isinstance(x, float) for x in emb))

        # Check normalization (L2 norm ≈ 1.0)
        norm = np.linalg.norm(emb)
        self.assertAlmostEqual(norm, 1.0, places=3)

        # Test chunk embedding
        chunk = TextChunk(
            chunk_id="test_live_01",
            text=text,
            doc_id="test_doc",
            chunk_index=0,
            metadata={"filename": "test.txt", "page": 1},
        )
        embedded_list = self.generator.embed_chunks([chunk])
        self.assertEqual(len(embedded_list), 1)
        self.assertEqual(embedded_list[0].chunk_id, "test_live_01")
        self.assertEqual(len(embedded_list[0].embedding), 384)

        mapping = self.generator.get_mapping(embedded_list)
        self.assertIn("test_live_01", mapping)
        self.assertEqual(mapping["test_live_01"]["text"], text)
        self.assertEqual(mapping["test_live_01"]["metadata"]["filename"], "test.txt")
        self.assertEqual(len(mapping["test_live_01"]["embedding"]), 384)


if __name__ == "__main__":
    unittest.main()
