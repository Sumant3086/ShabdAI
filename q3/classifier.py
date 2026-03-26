"""
Q3a/b — Hindi word spell checker with confidence scoring.
Classifies each word as 'correct spelling' or 'incorrect spelling'
with a confidence level (high/medium/low) and reason.
"""

import re
import unicodedata

# Valid Devanagari character range
_DEVANAGARI_CHAR = re.compile(r'^[\u0900-\u097F\u0966-\u096F\s]+$')
_DOUBLE_MATRA   = re.compile(r'[\u093E-\u094C]{2,}')
_DOUBLE_HALANT  = re.compile(r'\u094D{2,}')
_LEADING_MATRA  = re.compile(r'^[\u093E-\u094C\u094D]')

# English loanwords in Devanagari — always correct per transcription guidelines
KNOWN_LOANWORDS = {
    "प्रोजेक्ट", "इंटरव्यू", "जॉब", "प्रॉब्लम", "सॉल्व", "कंप्यूटर",
    "मोबाइल", "फोन", "इंटरनेट", "ऑनलाइन", "ऑफलाइन", "वेबसाइट",
    "टेंट", "कैम्प", "कैम्पिंग", "एरिया", "लाइट", "गार्ड",
    "बस", "ट्रेन", "कार", "बाइक", "स्कूल", "कॉलेज", "क्लास",
    "टीचर", "डॉक्टर", "नर्स", "हॉस्पिटल", "मार्केट", "शॉप",
    "होटल", "रेस्टोरेंट", "पार्क", "रोड", "स्टेशन", "अमेज़न",
    "मिस्टेक", "लैंड",
}

VALID_SUFFIXES = [
    "ना", "ता", "ती", "ते", "या", "यी", "ये", "ओ", "ए",
    "ाना", "ाती", "ाते", "ाया", "ाई", "ाए", "ेगा", "ेगी",
    "ेंगे", "ेंगी", "ूँगा", "ूँगी",
]


def normalise_word(word: str) -> str:
    word = unicodedata.normalize("NFC", word)
    return re.sub(r'^[^\u0900-\u097F]+|[^\u0900-\u097F]+$', '', word)


def is_valid_devanagari(word: str) -> bool:
    if not word or not _DEVANAGARI_CHAR.match(word):
        return False
    if _DOUBLE_MATRA.search(word) or _DOUBLE_HALANT.search(word):
        return False
    if _LEADING_MATRA.match(word):
        return False
    return True


def classify_word(word: str, dictionary: set, freq: int, total_words: int) -> dict:
    """
    Classify a word as correct/incorrect with confidence and reason.

    Returns dict with keys: word, label, confidence, reason
    """
    norm = normalise_word(word)

    if not norm:
        return {"word": word, "label": "incorrect spelling",
                "confidence": "high", "reason": "empty after normalisation"}

    if re.search(r'[a-zA-Z]', norm):
        return {"word": word, "label": "incorrect spelling",
                "confidence": "high",
                "reason": "contains Roman characters — should be Devanagari per guidelines"}

    if norm in KNOWN_LOANWORDS:
        return {"word": word, "label": "correct spelling",
                "confidence": "high", "reason": "known Devanagari loanword"}

    if not is_valid_devanagari(norm):
        return {"word": word, "label": "incorrect spelling",
                "confidence": "high", "reason": "invalid Devanagari character sequence"}

    if norm in dictionary:
        return {"word": word, "label": "correct spelling",
                "confidence": "high", "reason": "found in Hindi dictionary"}

    freq_ratio = freq / total_words if total_words > 0 else 0
    is_rare = freq_ratio < 1e-5

    if is_rare:
        return {"word": word, "label": "incorrect spelling",
                "confidence": "low",
                "reason": "not in dictionary and very rare — possible misspelling or proper noun"}

    if any(norm.endswith(s) for s in VALID_SUFFIXES):
        return {"word": word, "label": "correct spelling",
                "confidence": "medium",
                "reason": "not in base dictionary but has valid Hindi morphological suffix"}

    return {"word": word, "label": "incorrect spelling",
            "confidence": "low",
            "reason": "not in dictionary, moderate frequency — uncertain"}
