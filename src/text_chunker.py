"""
NLP Lab Experiment 5: Stage 4 - Text Chunking
=============================================
Provides sliding-window text chunking with configurable size and overlap,
boundary awareness, metadata preservation, and unique chunk IDs.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
import logging

from src.document_loader import Document

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """Represents a chunked unit of text derived from a parent Document.
    
    Attributes:
        chunk_id: Globally unique identifier for this chunk (e.g. 'doc1_c001').
        text: The cleaned, segmented text content of the chunk.
        doc_id: The identifier of the parent Document.
        chunk_index: 0-indexed position of this chunk within its parent Document.
        metadata: Inherited document metadata plus chunk positional information.
    """
    chunk_id: str
    text: str
    doc_id: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        snippet = (self.text[:50] + "...") if len(self.text) > 50 else self.text
        return (
            f"TextChunk(id={self.chunk_id!r}, "
            f"source={self.metadata.get('filename')!r}, "
            f"page={self.metadata.get('page')}, "
            f"chunk_idx={self.chunk_index}, "
            f"chars={self.metadata.get('char_count')}, "
            f"text={snippet!r})"
        )


class TextChunker:
    """Partitions Document objects into smaller, overlapping semantic chunks."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """Initializes the chunker with target size and overlap in characters.
        
        Args:
            chunk_size: Maximum target character count for each chunk (default: 500).
            chunk_overlap: Number of characters to share between adjacent chunks (default: 100).
        
        Raises:
            ValueError: If chunk_size <= 0 or chunk_overlap >= chunk_size.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap cannot be negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.step_size = chunk_size - chunk_overlap

    def _find_boundary(self, text: str, target_pos: int, window_radius: int = 50) -> int:
        """Finds a natural punctuation or whitespace boundary near target_pos to avoid cutting words."""
        if target_pos >= len(text):
            return len(text)

        search_start = max(0, target_pos - window_radius)
        search_end = min(len(text), target_pos + window_radius)
        window = text[search_start:search_end]

        # Priority 1: Paragraph break (\n\n)
        p_break = window.rfind("\n\n")
        if p_break != -1 and search_start + p_break > search_start:
            return search_start + p_break + 2

        # Priority 2: Sentence endings (. \n ? !)
        for punct in (". ", ".\n", "? ", "! "):
            punct_pos = window.rfind(punct)
            if punct_pos != -1:
                return search_start + punct_pos + len(punct)

        # Priority 3: Single newline or whitespace
        space_pos = window.rfind(" ")
        if space_pos != -1:
            return search_start + space_pos + 1

        # Fallback: strictly at target_pos if no boundary found
        return target_pos

    def _find_word_start(self, text: str, pos: int) -> int:
        """Snaps pos to the start of the word so chunk never begins with a clipped word."""
        if pos <= 0:
            return 0
        if pos >= len(text):
            return len(text)
        # If previous character is whitespace, pos is already at the start of a word
        if text[pos - 1].isspace():
            return pos
        # Walk backwards to find the start of the current word
        while pos > 0 and not text[pos - 1].isspace():
            pos -= 1
        return pos

    def split_text(self, text: str) -> List[Tuple[str, int, int]]:
        """Splits raw string into overlapping segments with boundary awareness.
        
        Returns:
            List of (chunk_text, start_char, end_char) tuples.
        """
        cleaned = text.strip()
        if not cleaned:
            return []

        # If text is smaller than chunk size, return it directly as a single chunk
        if len(cleaned) <= self.chunk_size:
            return [(cleaned, 0, len(cleaned))]

        chunks: List[Tuple[str, int, int]] = []
        start = 0
        total_len = len(cleaned)

        while start < total_len:
            raw_end = start + self.chunk_size
            if raw_end >= total_len:
                end = total_len
            else:
                end = self._find_boundary(cleaned, raw_end)

            chunk_text = cleaned[start:end].strip()

            # Ensure we do not add empty chunks
            if chunk_text:
                chunks.append((chunk_text, start, end))

            if end >= total_len:
                break

            # Calculate next start position, incorporating overlap
            raw_next_start = end - self.chunk_overlap
            next_start = self._find_word_start(cleaned, max(0, raw_next_start))

            # Guard against infinite loops where step cannot advance
            if next_start <= start:
                next_start = start + max(1, self.step_size)

            start = next_start

        return chunks

    def chunk_document(self, doc: Document) -> List[TextChunk]:
        """Splits a single Document into a list of TextChunk objects with preserved metadata.
        
        Args:
            doc: Input Document instance.

        Returns:
            List of TextChunk instances.
        """
        raw_chunks = self.split_text(doc.text)
        text_chunks: List[TextChunk] = []

        for idx, (chunk_text, start_char, end_char) in enumerate(raw_chunks):
            # Combine parent document metadata with chunk-specific fields
            chunk_metadata = dict(doc.metadata)
            chunk_metadata.update({
                "parent_doc_id": doc.doc_id,
                "chunk_index": idx,
                "total_chunks_in_doc": len(raw_chunks),
                "start_char": start_char,
                "end_char": end_char,
                "char_count": len(chunk_text),
                "word_count": len(chunk_text.split()),
                "chunk_size_setting": self.chunk_size,
                "chunk_overlap_setting": self.chunk_overlap,
            })

            # Create unique deterministic chunk ID
            chunk_id = f"{doc.doc_id}_c{idx:03d}"

            text_chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    doc_id=doc.doc_id,
                    chunk_index=idx,
                    metadata=chunk_metadata,
                )
            )

        return text_chunks

    def chunk_documents(self, documents: List[Document]) -> List[TextChunk]:
        """Processes a collection of Document objects into a flat list of TextChunks.
        
        Args:
            documents: List of Document instances.

        Returns:
            Flat list of all generated TextChunk instances.
        """
        all_chunks: List[TextChunk] = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)

        return all_chunks
