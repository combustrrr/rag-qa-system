"""
NLP Lab Experiment 5: Stages 1 & 2 - Document Loading and Text Extraction
========================================================================
Handles file discovery, format-specific extraction (.pdf, .txt),
sanitization, and standardized document packaging.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging
import hashlib

from src.text_cleaner import clean_text

# Configure module logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


@dataclass
class Document:
    """Represents an extracted unit of text along with its provenance metadata.
    
    Attributes:
        doc_id: Unique deterministic or sequential identifier.
        text: The cleaned text content of the document.
        metadata: Associated contextual metadata (source, page, word count, etc.).
    """
    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        snippet = (self.text[:60] + "...") if len(self.text) > 60 else self.text
        return (
            f"Document(id={self.doc_id!r}, "
            f"source={self.metadata.get('filename')!r}, "
            f"page={self.metadata.get('page')}, "
            f"words={self.metadata.get('word_count')}, "
            f"text={snippet!r})"
        )


class DocumentLoader:
    """Discovers, reads, extracts, and normalizes documents from the filesystem."""

    SUPPORTED_EXTENSIONS = {".txt", ".pdf"}

    def __init__(self, data_dir: Optional[Path | str] = None):
        """Initializes the loader with an optional base data directory.
        
        Args:
            data_dir: Path to directory containing documents.
        """
        self.data_dir = Path(data_dir).resolve() if data_dir else None

    @staticmethod
    def _generate_doc_id(source_name: str, page_number: int, content: str) -> str:
        """Generates a stable, unique ID based on file, page, and content hash."""
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:8]
        return f"{source_name}_p{page_number}_{content_hash}"

    def discover_files(self, directory: Optional[Path | str] = None, recursive: bool = True) -> List[Path]:
        """Discovers files in the target directory and filters supported extensions.
        
        Args:
            directory: Directory to scan. Defaults to self.data_dir.
            recursive: Whether to scan subdirectories.

        Returns:
            List of valid, supported file paths.
        """
        target_dir = Path(directory).resolve() if directory else self.data_dir
        if not target_dir or not target_dir.exists():
            logger.warning(f"Directory not found: {target_dir}")
            return []

        pattern = "**/*" if recursive else "*"
        all_entries = [p for p in target_dir.glob(pattern) if p.is_file() and not p.name.startswith(".")]

        valid_files = []
        for file_path in all_entries:
            ext = file_path.suffix.lower()
            if ext not in self.SUPPORTED_EXTENSIONS:
                logger.warning(
                    f"Skipping unsupported file: '{file_path.name}' "
                    f"(supported extensions: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))})"
                )
                continue

            if file_path.stat().st_size == 0:
                logger.warning(f"Skipping empty file (0 bytes): '{file_path.name}'")
                continue

            valid_files.append(file_path)

        return sorted(valid_files)

    def _extract_txt(self, file_path: Path) -> List[Document]:
        """Extracts and cleans text from a plain text (.txt) file."""
        content = ""
        # Try UTF-8 first, fallback to latin-1 for legacy encodings
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, "r", encoding="latin-1") as f:
                    content = f.read()
            except Exception as e:
                logger.error(f"Failed to read text file '{file_path.name}': {e}")
                return []
        except Exception as e:
            logger.error(f"Error opening text file '{file_path.name}': {e}")
            return []

        cleaned = clean_text(content)
        if not cleaned:
            logger.warning(f"File '{file_path.name}' produced no extractable text after cleaning.")
            return []

        doc_id = self._generate_doc_id(file_path.stem, 1, cleaned)
        words = len(cleaned.split())
        metadata = {
            "source": str(file_path),
            "filename": file_path.name,
            "file_type": "txt",
            "page": 1,
            "total_pages": 1,
            "char_count": len(cleaned),
            "word_count": words,
        }
        return [Document(doc_id=doc_id, text=cleaned, metadata=metadata)]

    def _extract_pdf(self, file_path: Path) -> List[Document]:
        """Extracts and cleans text page-by-page from a PDF (.pdf) file using pypdf."""
        try:
            from pypdf import PdfReader
        except ImportError:
            logger.error("pypdf is required for PDF text extraction. Install it with: pip install pypdf")
            return []

        documents = []
        try:
            reader = PdfReader(str(file_path))
            total_pages = len(reader.pages)

            if total_pages == 0:
                logger.warning(f"PDF file '{file_path.name}' contains 0 pages.")
                return []

            for page_idx, page in enumerate(reader.pages, start=1):
                raw_page_text = page.extract_text() or ""
                cleaned = clean_text(raw_page_text)

                if not cleaned:
                    logger.info(f"Page {page_idx} of '{file_path.name}' has no extractable text (might be scanned or blank).")
                    continue

                doc_id = self._generate_doc_id(file_path.stem, page_idx, cleaned)
                words = len(cleaned.split())
                metadata = {
                    "source": str(file_path),
                    "filename": file_path.name,
                    "file_type": "pdf",
                    "page": page_idx,
                    "total_pages": total_pages,
                    "char_count": len(cleaned),
                    "word_count": words,
                }
                documents.append(Document(doc_id=doc_id, text=cleaned, metadata=metadata))

            if not documents:
                logger.warning(f"PDF '{file_path.name}' produced no readable text across all {total_pages} page(s).")

        except Exception as e:
            logger.error(f"Failed to extract PDF '{file_path.name}': {e}")
            return []

        return documents

    def load_file(self, file_path: Path | str) -> List[Document]:
        """Loads and extracts text from a single file path (.txt or .pdf).
        
        Args:
            file_path: Path to target document.

        Returns:
            List of Document objects extracted from the file.
        """
        path = Path(file_path).resolve()
        if not path.is_file():
            logger.warning(f"File does not exist: {path}")
            return []

        if path.stat().st_size == 0:
            logger.warning(f"Skipping empty file (0 bytes): '{path.name}'")
            return []

        ext = path.suffix.lower()
        if ext == ".txt":
            return self._extract_txt(path)
        elif ext == ".pdf":
            return self._extract_pdf(path)
        else:
            logger.warning(
                f"Skipping unsupported file extension: '{path.name}' "
                f"(supported: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))})"
            )
            return []

    def load_directory(self, directory: Optional[Path | str] = None, recursive: bool = True) -> List[Document]:
        """Discovers, parses, and extracts all supported documents in a directory.
        
        Args:
            directory: Directory to process. Defaults to self.data_dir.
            recursive: Whether to scan subdirectories.

        Returns:
            List of standardized Document objects.
        """
        files = self.discover_files(directory=directory, recursive=recursive)
        if not files:
            logger.info("No valid document files discovered to process.")
            return []

        all_documents: List[Document] = []
        for file_path in files:
            docs = self.load_file(file_path)
            all_documents.extend(docs)

        return all_documents
