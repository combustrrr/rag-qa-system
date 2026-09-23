"""
NLP Lab Experiment 5: Stage 5 - Embedding Generation
====================================================
Transforms text chunks generated in Stage 4 into dense semantic vector
embeddings using a lightweight, open-source sentence-transformers model.

Features:
- Encapsulates embedding logic in EmbeddingGenerator
- Loads the model into memory exactly once
- Maintains the explicit mapping: chunk_id -> text -> metadata -> embedding
- Simple, pedagogical design tailored for an NLP laboratory practical
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging

from src.document_loader import DocumentLoader
from src.text_chunker import TextChunker, TextChunk

logger = logging.getLogger(__name__)

# Default lightweight model: fast, CPU-friendly, 384 dimensions (~80MB)
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


@dataclass
class EmbeddedChunk:
    """Represents a text chunk paired with its dense vector embedding.

    Attributes:
        chunk_id: Unique identifier for the chunk (e.g., 'doc1_p1_c000').
        text: The cleaned, segmented text content of the chunk.
        metadata: Inherited document metadata plus chunk positional information.
        embedding: Dense numerical vector representation (List of floats).
    """
    chunk_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: List[float] = field(default_factory=list)

    @property
    def embedding_dim(self) -> int:
        """Returns the dimensionality of the embedding vector."""
        return len(self.embedding)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the embedded chunk into a dictionary mapping:
        text, metadata, and embedding.
        """
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata,
            "embedding": self.embedding,
        }

    def __repr__(self) -> str:
        snippet = (self.text[:50] + "...") if len(self.text) > 50 else self.text
        return (
            f"EmbeddedChunk(id={self.chunk_id!r}, "
            f"dim={self.embedding_dim}, "
            f"source={self.metadata.get('filename')!r}, "
            f"text={snippet!r})"
        )


class EmbeddingGenerator:
    """Loads a sentence embedding model once and converts TextChunk objects

    into dense numerical vectors.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: Optional[str] = None):
        """Loads the pre-trained embedding model into memory once.

        Args:
            model_name: Hugging Face model identifier (default: 'all-MiniLM-L6-v2').
            device: Target device ('cpu', 'cuda', or None for automatic selection).
        """
        self.model_name = model_name
        self.device = device

        logger.info(f"Loading embedding model '{self.model_name}' (device={self.device})...")

        # Lazy import to ensure fast module import if dependencies are missing during checks
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for Stage 5 Embedding Generation. "
                "Install it using: pip install sentence-transformers"
            ) from exc

        # Load model into memory ONCE
        self._model = SentenceTransformer(self.model_name, device=self.device)
        if hasattr(self._model, "get_embedding_dimension"):
            self._embedding_dim = self._model.get_embedding_dimension()
        else:
            self._embedding_dim = self._model.get_sentence_embedding_dimension()
        logger.info(f"Model '{self.model_name}' loaded successfully (dimension={self._embedding_dim}).")

    @property
    def embedding_dimension(self) -> int:
        """Dimensionality of the dense vectors produced by this model."""
        return self._embedding_dim

    def embed_text(self, text: str) -> List[float]:
        """Encodes a single string into a 1D embedding vector.

        Args:
            text: Input string (e.g., query or single chunk text).

        Returns:
            Dense vector as a list of floats.
        """
        if not text or not text.strip():
            # Return zero vector for empty strings
            return [0.0] * self._embedding_dim

        vector = self._model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return vector.tolist()

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Batch encodes multiple strings into dense embedding vectors.

        Args:
            texts: List of strings to encode.
            batch_size: Number of texts to encode per batch.

        Returns:
            List of embedding vectors (each vector is a List[float]).
        """
        if not texts:
            return []

        # Replace completely blank texts with a space to prevent encode anomalies
        clean_texts = [t if (t and t.strip()) else " " for t in texts]

        vectors = self._model.encode(
            clean_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return vectors.tolist()

    def embed_chunks(self, chunks: List[TextChunk], batch_size: int = 32) -> List[EmbeddedChunk]:
        """Converts a collection of TextChunk objects into EmbeddedChunk objects,
        preserving chunk_id, text, metadata, and adding the dense embedding vector.

        Args:
            chunks: List of TextChunk objects from Stage 4.
            batch_size: Batch size for model inference.

        Returns:
            List of EmbeddedChunk instances.
        """
        if not chunks:
            return []

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embed_texts(texts, batch_size=batch_size)

        embedded_chunks: List[EmbeddedChunk] = []
        for chunk, emb in zip(chunks, embeddings):
            # Maintain explicit metadata with embedding-specific details
            metadata = dict(chunk.metadata)
            metadata["embedding_model"] = self.model_name
            metadata["embedding_dim"] = len(emb)

            embedded_chunks.append(
                EmbeddedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    metadata=metadata,
                    embedding=emb,
                )
            )

        return embedded_chunks

    @staticmethod
    def get_mapping(embedded_chunks: List[EmbeddedChunk]) -> Dict[str, Dict[str, Any]]:
        """Maintains and returns the explicit mapping:
        chunk_id -> text -> metadata -> embedding

        Returns:
            Dictionary mapping chunk_id to its text, metadata, and embedding.
        """
        mapping: Dict[str, Dict[str, Any]] = {}
        for ec in embedded_chunks:
            mapping[ec.chunk_id] = {
                "text": ec.text,
                "metadata": ec.metadata,
                "embedding": ec.embedding,
            }
        return mapping


def print_embedding_theory():
    """Prints educational theoretical concepts regarding dense sentence embeddings."""
    print("=" * 80)
    print("   THEORETICAL INSIGHT: DENSE EMBEDDINGS VS. SPARSE KEYWORD SEARCH")
    print("=" * 80)
    print("1. Why Dense Semantic Vectors:")
    print("   • Semantic Understanding: Traditional Bag-of-Words or TF-IDF rely on exact word")
    print("                             matches. Dense embeddings capture meaning, synonyms,")
    print("                             and conceptual relationships (e.g. 'doctor' ~ 'physician').")
    print("   • Continuous Vector Space: Text is mapped to a continuous 384-dimensional hypersphere")
    print("                             where cosine similarity reflects contextual closeness.")
    print("\n2. Vector Mapping in RAG:")
    print("   • chunk_id   : Deterministic identifier for traceable retrieval citations.")
    print("   • text       : Contextual passage to be passed to LLM prompt in Stage 8.")
    print("   • metadata   : Page number, file type, source used for filtering & attribution.")
    print("   • embedding  : 384 float numbers ready to index in ChromaDB (Stage 6).")
    print("=" * 80 + "\n")


def run_embedding_generation(
    data_dir: Optional[Path] = None,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    model_name: str = DEFAULT_MODEL_NAME,
) -> Tuple[List[EmbeddedChunk], Dict[str, Dict[str, Any]]]:
    """Executes Stage 5: End-to-end Embedding Generation.

    1. Loads and extracts documents from data_dir (Stages 1-3).
    2. Chunks documents into overlapping segments (Stage 4).
    3. Loads the sentence embedding model ONCE (Stage 5).
    4. Generates dense vector embeddings for all chunks.
    5. Constructs and validates the chunk_id -> text -> metadata -> embedding mapping.
    6. Displays the required summary and example chunk with embedding shape.

    Returns:
        Tuple of (List[EmbeddedChunk], Dict[chunk_id -> {text, metadata, embedding}]).
    """
    if data_dir is None:
        base_dir = Path(__file__).resolve().parent.parent
        data_dir = base_dir / "data" / "sample_docs"

    print("=" * 80)
    print("   STAGE 5: EMBEDDING GENERATION PIPELINE")
    print("=" * 80)
    print(f"[*] Target Directory  : {data_dir}")
    print(f"[*] Chunk Size/Overlap: {chunk_size} chars / {chunk_overlap} chars")
    print(f"[*] Loading Model     : {model_name} (loading ONCE into memory)...\n")

    # Step A: Load and Chunk Documents
    loader = DocumentLoader(data_dir)
    documents = loader.load_directory()
    if not documents:
        print("[!] No documents found to process. Please check data directory.")
        return [], {}

    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.chunk_documents(documents)
    if not chunks:
        print("[!] No text chunks were generated.")
        return [], {}

    # Step B: Load Embedding Generator ONCE
    embedder = EmbeddingGenerator(model_name=model_name)

    # Step C: Generate Embeddings for Every Chunk
    print(f"[*] Generating dense embeddings for {len(chunks)} chunk(s)...")
    embedded_chunks = embedder.embed_chunks(chunks)

    # Step D: Maintain the chunk_id -> text -> metadata -> embedding mapping
    mapping = embedder.get_mapping(embedded_chunks)

    # Step E: Print required laboratory outputs
    print("\n" + "=" * 80)
    print("   EMBEDDING GENERATION RESULTS")
    print("=" * 80)
    print(f" • Embedding Model Name : {embedder.model_name}")
    print(f" • Number of Chunks     : {len(embedded_chunks)}")
    print(f" • Embedding Dimension  : {embedder.embedding_dimension}")
    print("-" * 80)

    # Step F: Print one example chunk and its embedding shape
    if embedded_chunks:
        example = embedded_chunks[0]
        meta = example.metadata
        print("[+] Example Chunk Representation:")
        print(f"  • Chunk ID        : {example.chunk_id}")
        print(f"  • Parent Doc ID   : {meta.get('parent_doc_id')}")
        print(f"  • Source Filename : {meta.get('filename')}")
        print(f"  • Page            : {meta.get('page')} of {meta.get('total_pages')}")
        print(f"  • Text Snippet    : \"{example.text[:80]}...\"")
        print(f"  • Embedding Shape : ({len(example.embedding)},)")
        print(f"  • Vector Preview  : [{', '.join(f'{x:.4f}' for x in example.embedding[:5])}, ...]")
        print("-" * 80)

    # Verify mapping integrity
    sample_id = embedded_chunks[0].chunk_id
    sample_entry = mapping[sample_id]
    assert "text" in sample_entry and "metadata" in sample_entry and "embedding" in sample_entry, "Mapping invalid!"
    print(f"[✓] Mapping Verified    : chunk_id → text → metadata → embedding ({len(mapping)} entries maintained)")
    print()

    print_embedding_theory()
    print(f"[✓] Stage 5 complete! {len(embedded_chunks)} embedded chunks ready for Vector Database indexing (Stage 6).\n")
    return embedded_chunks, mapping


if __name__ == "__main__":
    run_embedding_generation()
