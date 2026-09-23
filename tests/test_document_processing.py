"""
Unit tests for Document Processing (Stages 1-3).
"""

import unittest
from pathlib import Path

from src.text_cleaner import clean_text
from src.document_loader import DocumentLoader, Document

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_DIR = BASE_DIR / "data" / "sample_docs"


class TestTextCleaner(unittest.TestCase):
    def test_unicode_and_whitespace(self):
        raw = "Hello   world!\r\n\r\nThis   is  a\ttest.\n\n\n\nNew paragraph."
        cleaned = clean_text(raw)
        self.assertNotIn("\r", cleaned)
        self.assertNotIn("   ", cleaned)
        self.assertNotIn("\n\n\n", cleaned)
        self.assertIn("Hello world!", cleaned)
        self.assertIn("This is a test.", cleaned)
        self.assertIn("New paragraph.", cleaned)

    def test_hyphenation_fix(self):
        raw = "The trans-\nformer architecture and self-\nattention mechanisms."
        cleaned = clean_text(raw)
        self.assertIn("transformer", cleaned)
        self.assertIn("selfattention", cleaned)

    def test_empty_string(self):
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text("   \n\t  "), "")


class TestDocumentLoader(unittest.TestCase):
    def setUp(self):
        self.loader = DocumentLoader(SAMPLE_DIR)

    def test_file_discovery_filters_empty_and_unsupported(self):
        discovered = self.loader.discover_files()
        filenames = [f.name for f in discovered]

        # Supported non-empty files must be present
        self.assertIn("attention_intro.txt", filenames)
        self.assertIn("sample_rag.pdf", filenames)

        # 0-byte and unsupported files must be filtered out
        self.assertNotIn("empty_sample.txt", filenames)
        self.assertNotIn("unsupported_doc.png", filenames)

    def test_txt_extraction_and_metadata(self):
        txt_path = SAMPLE_DIR / "attention_intro.txt"
        docs = self.loader.load_file(txt_path)

        self.assertEqual(len(docs), 1)
        doc = docs[0]
        self.assertIsInstance(doc, Document)
        self.assertTrue(doc.doc_id.startswith("attention_intro_p1_"))
        self.assertEqual(doc.metadata["filename"], "attention_intro.txt")
        self.assertEqual(doc.metadata["file_type"], "txt")
        self.assertEqual(doc.metadata["page"], 1)
        self.assertEqual(doc.metadata["total_pages"], 1)
        self.assertGreater(doc.metadata["word_count"], 50)
        self.assertIn("Attention Is All You Need", doc.text)

    def test_pdf_extraction_and_pages(self):
        pdf_path = SAMPLE_DIR / "sample_rag.pdf"
        docs = self.loader.load_file(pdf_path)

        self.assertEqual(len(docs), 2)

        # Page 1 checks
        doc1 = docs[0]
        self.assertEqual(doc1.metadata["filename"], "sample_rag.pdf")
        self.assertEqual(doc1.metadata["page"], 1)
        self.assertEqual(doc1.metadata["total_pages"], 2)
        self.assertIn("Retrieval-Augmented Generation", doc1.text)

        # Page 2 checks
        doc2 = docs[1]
        self.assertEqual(doc2.metadata["page"], 2)
        self.assertEqual(doc2.metadata["total_pages"], 2)
        self.assertIn("Vector Databases", doc2.text)

    def test_load_directory_complete(self):
        all_docs = self.loader.load_directory()
        # 1 from txt + 2 from pdf = 3 valid documents
        self.assertGreaterEqual(len(all_docs), 3)
        for doc in all_docs:
            self.assertTrue(doc.doc_id)
            self.assertTrue(doc.text)
            self.assertIn("filename", doc.metadata)
            self.assertIn("page", doc.metadata)


if __name__ == "__main__":
    unittest.main()
