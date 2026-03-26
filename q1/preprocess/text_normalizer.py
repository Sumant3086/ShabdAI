"""
Q1a — Text normalisation for Hindi ASR transcripts.
Strips punctuation, applies NFC unicode, collapses whitespace.
Per guidelines: English loanwords in Devanagari are kept as-is.
"""

import re
import unicodedata

# Punctuation to strip (danda, double-danda, common marks)
_STRIP_CHARS = re.compile(r'[।|॥,\.\!\?\"\'\(\)\[\]\{\};:—–\-]')
_MULTI_SPACE = re.compile(r'\s+')


def normalize_text(text: str) -> str:
    """
    Light normalisation pipeline:
    1. NFC unicode normalisation
    2. Strip danda / punctuation
    3. Collapse whitespace
    4. Strip leading/trailing space

    Intentionally preserves:
    - Devanagari digits
    - English loanwords written in Devanagari (per transcription guidelines)
    - Anusvara and chandrabindu (critical for Hindi morphology)
    """
    text = unicodedata.normalize("NFC", text)
    text = _STRIP_CHARS.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


def extract_text_from_transcription(trans_data) -> str:
    """
    Extract plain text from transcription JSON.
    Handles both list-of-segments and dict formats.
    """
    if isinstance(trans_data, list):
        return " ".join(seg.get("text", "") for seg in trans_data)
    elif isinstance(trans_data, dict):
        return trans_data.get("text", trans_data.get("transcription", ""))
    return str(trans_data)
