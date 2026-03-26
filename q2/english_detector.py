"""
Q2b — English word detector for Hindi ASR transcripts.
Tags Roman-script tokens and known Devanagari loanwords with [EN][/EN].

Per transcription guidelines: English words spoken in conversation are
transcribed in Devanagari. So "computer" -> "कंप्यूटर" is CORRECT.
This module identifies both cases for downstream processing.
"""

import re

# Known English loanwords written in Devanagari (correct per guidelines)
DEVANAGARI_LOANWORDS = {
    "प्रोजेक्ट", "इंटरव्यू", "जॉब", "प्रॉब्लम", "सॉल्व", "कंप्यूटर",
    "मोबाइल", "फोन", "इंटरनेट", "ऑनलाइन", "ऑफलाइन", "वेबसाइट",
    "टेंट", "कैम्प", "कैम्पिंग", "एरिया", "लाइट", "गार्ड",
    "बस", "ट्रेन", "कार", "बाइक", "स्कूल", "कॉलेज", "क्लास",
    "टीचर", "डॉक्टर", "नर्स", "हॉस्पिटल", "मार्केट", "शॉप",
    "होटल", "रेस्टोरेंट", "पार्क", "रोड", "स्टेशन", "अमेज़न",
    "मिस्टेक", "लैंड", "जंगल",
}

_ROMAN_WORD = re.compile(r'\b[a-zA-Z]{2,}\b')


def tag_english_words(text: str) -> str:
    """
    Tag English words in a Hindi transcript with [EN]...[/EN].

    Handles:
    1. Roman-script tokens (model output Roman instead of Devanagari)
    2. Known English loanwords written in Devanagari

    Examples:
      "मेरा interview अच्छा गया"
      -> "मेरा [EN]interview[/EN] अच्छा गया"

      "मेरा इंटरव्यू बहुत अच्छा गया और मुझे जॉब मिल गई"
      -> "मेरा [EN]इंटरव्यू[/EN] बहुत अच्छा गया और मुझे [EN]जॉब[/EN] मिल गई"
    """
    # Tag Roman-script tokens first
    tagged = _ROMAN_WORD.sub(lambda m: f"[EN]{m.group(0)}[/EN]", text)

    # Tag known Devanagari loanwords
    words = tagged.split()
    result = []
    for word in words:
        if "[EN]" in word:
            result.append(word); continue
        clean = re.sub(r'[^\u0900-\u097F]', '', word)
        if clean and clean in DEVANAGARI_LOANWORDS:
            result.append(f"[EN]{word}[/EN]")
        else:
            result.append(word)

    return " ".join(result)


def get_english_words(text: str) -> list:
    """Return list of detected English words (both Roman and Devanagari loanwords)."""
    tagged = tag_english_words(text)
    # Use escaped brackets so [EN] and [/EN] are treated as literals
    return re.findall(r'\[EN\](.+?)\[/EN\]', tagged)
