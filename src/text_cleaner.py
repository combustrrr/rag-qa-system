"""
NLP Lab Experiment 5: Stage 3 - Text Cleaning and Sanitization
==============================================================
Provides sanitization and normalization for raw text extracted from documents.
"""

import re
import unicodedata


def clean_text(raw_text: str) -> str:
    """Cleans and normalizes raw text extracted from documents.
    
    Operations performed:
    1. Unicode normalization (NFKC) to standardize characters (e.g. curly quotes, ligatures).
    2. Normalize line endings (\\r\\n and \\r to \\n).
    3. Reconstruct hyphenated words split across lines (e.g., 'trans-\\nformer' -> 'transformer').
    4. Strip non-printable control characters while preserving standard whitespace (\\n, \\t).
    5. Collapse multiple horizontal whitespaces/tabs into a single space.
    6. Collapse 3 or more consecutive newlines into 2 to preserve clean paragraph breaks.
    7. Strip leading and trailing whitespace.

    Args:
        raw_text: Raw string input from files.

    Returns:
        Cleaned, normalized string.
    """
    if not raw_text:
        return ""

    # 1. Unicode normalization
    text = unicodedata.normalize("NFKC", raw_text)

    # 2. Normalize carriage returns to standard newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 3. Fix words hyphenated across line breaks (e.g., 'knowl-\nedge' -> 'knowledge')
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)

    # 4. Remove non-printable control characters, keeping newlines, carriage returns, tabs
    text = "".join(ch for ch in text if ch.isprintable() or ch in ("\n", "\t"))

    # 5. Collapse consecutive horizontal whitespace (spaces/tabs) on each line
    text = re.sub(r'[ \t]+', ' ', text)

    # 6. Normalize multiple newlines (3 or more -> 2)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 7. Strip whitespace at the start and end of the entire string
    return text.strip()
