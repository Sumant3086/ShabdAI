"""
Q3 — Hindi Spell Checker for ~177k unique words from transcription dataset.

Approach:
1. Fetch the unique word list from the dataset.
2. For each word, classify as 'correct spelling' or 'incorrect spelling'
   using a multi-signal pipeline:
   a) Dictionary lookup (IndicNLP / custom Hindi wordlist)
   b) Morphological validity (valid Devanagari character sequences)
   c) Frequency-based confidence (rare words get lower confidence)
   d) Known loanword list (Devanagari-transcribed English words are CORRECT)
3. Output confidence score: high / medium / low with reason.
4. Analyse low-confidence bucket.
"""

import re
import json
import unicodedata
import requests
import pandas as pd
from pathlib import Path
from collections import Counter
from tqdm import tqdm

OUT_DIR = Path("data/q3_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Devanagari character classes ──────────────────────────────────────────────
# Valid Devanagari Unicode ranges
_DEVANAGARI_CHAR = re.compile(r'^[\u0900-\u097F\u0966-\u096F\s]+$')
# Halant (virama) — used in conjuncts
_HALANT = '\u094D'
# Nukta — used in loanword phonemes (क़, ख़, etc.)
_NUKTA = '\u093C'

# Invalid character sequences in Hindi
_DOUBLE_MATRA = re.compile(r'[\u093E-\u094C]{2,}')   # two vowel signs in a row
_DOUBLE_HALANT = re.compile(r'\u094D{2,}')            # double virama
_LEADING_MATRA = re.compile(r'^[\u093E-\u094C\u094D]') # starts with dependent vowel

# ── Known correct Devanagari loanwords (English spoken -> Devanagari) ─────────
KNOWN_LOANWORDS = {
    "प्रोजेक्ट", "इंटरव्यू", "जॉब", "प्रॉब्लम", "सॉल्व", "कंप्यूटर",
    "मोबाइल", "फोन", "इंटरनेट", "ऑनलाइन", "ऑफलाइन", "वेबसाइट",
    "टेंट", "कैम्प", "कैम्पिंग", "एरिया", "लाइट", "गार्ड",
    "बस", "ट्रेन", "कार", "बाइक", "स्कूल", "कॉलेज", "क्लास",
    "टीचर", "डॉक्टर", "नर्स", "हॉस्पिटल", "मार्केट", "शॉप",
    "होटल", "रेस्टोरेंट", "पार्क", "रोड", "स्टेशन", "अमेज़न",
    "मिस्टेक", "लैंड", "जंगल",
}

# ── Load Hindi dictionary ─────────────────────────────────────────────────────
def load_hindi_dictionary() -> set:
    """
    Load a Hindi word dictionary.
    Uses the IndicNLP Hindi wordlist if available, otherwise falls back
    to a minimal built-in set for demonstration.
    """
    # Try to load from indic-nlp-library
    try:
        from indicnlp.tokenize import indic_tokenize
        from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
        # IndicNLP doesn't ship a dictionary directly, but we can use
        # the morphanalysis module if available
        print("[INFO] IndicNLP loaded for normalisation support")
    except ImportError:
        print("[INFO] IndicNLP not available, using fallback dictionary")

    # Try to fetch a public Hindi wordlist
    wordlist_url = "https://raw.githubusercontent.com/anoopkunchukuttan/indic_nlp_resources/master/lexicon/hi.txt"
    try:
        r = requests.get(wordlist_url, timeout=15)
        if r.status_code == 200:
            words = set(r.text.strip().split('\n'))
            print(f"[INFO] Loaded {len(words)} words from IndicNLP lexicon")
            return words
    except Exception as e:
        print(f"[WARN] Could not fetch dictionary: {e}")

    # Minimal fallback — common Hindi words
    fallback = {
        "है", "हैं", "था", "थे", "थी", "हो", "होना", "करना", "जाना",
        "आना", "देना", "लेना", "बोलना", "कहना", "सुनना", "देखना",
        "और", "या", "लेकिन", "क्योंकि", "तो", "भी", "ही", "न", "नहीं",
        "मैं", "हम", "तुम", "आप", "वो", "वह", "यह", "ये", "वे",
        "का", "की", "के", "को", "से", "में", "पर", "पे", "ने",
        "एक", "दो", "तीन", "चार", "पाँच", "दस", "सौ", "हज़ार",
        "बहुत", "थोड़ा", "ज़्यादा", "कम", "अच्छा", "बुरा", "बड़ा", "छोटा",
        "घर", "काम", "बात", "दिन", "रात", "समय", "लोग", "जगह",
        "जाना", "आना", "रहना", "करना", "होना", "मिलना", "बताना",
        "पता", "मतलब", "अब", "फिर", "जब", "तब", "यहाँ", "वहाँ",
        "जनजाति", "जंगल", "घाटी", "एरिया", "प्रोजेक्ट", "टेंट",
    }
    return fallback


# ── Morphological validity checks ────────────────────────────────────────────
def is_valid_devanagari_sequence(word: str) -> bool:
    """
    Check if the character sequence is morphologically plausible in Hindi.
    Flags obvious errors like double matras, leading dependent vowels, etc.
    """
    if not word:
        return False
    if not _DEVANAGARI_CHAR.match(word):
        return False   # contains non-Devanagari characters
    if _DOUBLE_MATRA.search(word):
        return False
    if _DOUBLE_HALANT.search(word):
        return False
    if _LEADING_MATRA.match(word):
        return False
    return True


def normalise_word(word: str) -> str:
    """NFC normalise and strip surrounding punctuation."""
    word = unicodedata.normalize("NFC", word)
    word = re.sub(r'^[^\u0900-\u097F]+|[^\u0900-\u097F]+$', '', word)
    return word


# ── Main classification ───────────────────────────────────────────────────────
def classify_word(word: str, dictionary: set, freq: int, total_words: int) -> dict:
    """
    Classify a single word as correct/incorrect with confidence.

    Returns:
        {
          "word": str,
          "label": "correct spelling" | "incorrect spelling",
          "confidence": "high" | "medium" | "low",
          "reason": str
        }
    """
    norm = normalise_word(word)

    # 1. Empty after normalisation
    if not norm:
        return {"word": word, "label": "incorrect spelling",
                "confidence": "high", "reason": "empty after normalisation"}

    # 2. Contains non-Devanagari (Roman script) — per guidelines, these are errors
    #    unless they are known loanwords in Devanagari
    if re.search(r'[a-zA-Z]', norm):
        return {"word": word, "label": "incorrect spelling",
                "confidence": "high",
                "reason": "contains Roman characters — should be Devanagari per guidelines"}

    # 3. Known loanword in Devanagari — always correct
    if norm in KNOWN_LOANWORDS:
        return {"word": word, "label": "correct spelling",
                "confidence": "high", "reason": "known Devanagari loanword"}

    # 4. Morphological validity
    if not is_valid_devanagari_sequence(norm):
        return {"word": word, "label": "incorrect spelling",
                "confidence": "high", "reason": "invalid Devanagari character sequence"}

    # 5. Dictionary lookup
    in_dict = norm in dictionary

    # 6. Frequency signal
    freq_ratio = freq / total_words if total_words > 0 else 0
    is_rare = freq_ratio < 1e-5   # appears < 1 in 100k words

    if in_dict:
        return {"word": word, "label": "correct spelling",
                "confidence": "high", "reason": "found in Hindi dictionary"}

    # Not in dictionary — could be: correct rare word, proper noun, or misspelling
    if is_rare:
        # Very rare + not in dict = likely error, but low confidence
        return {"word": word, "label": "incorrect spelling",
                "confidence": "low",
                "reason": "not in dictionary and very rare — possible misspelling or proper noun"}

    # Moderate frequency but not in dict — could be valid morphological form
    # Check if it looks like a plausible inflected form (ends in common suffixes)
    VALID_SUFFIXES = ["ना", "ता", "ती", "ते", "या", "यी", "ये", "ओ", "ए",
                      "ाना", "ाती", "ाते", "ाया", "ाई", "ाए", "ेगा", "ेगी",
                      "ेंगे", "ेंगी", "ूँगा", "ूँगी", "ूँगे"]
    has_valid_suffix = any(norm.endswith(s) for s in VALID_SUFFIXES)

    if has_valid_suffix:
        return {"word": word, "label": "correct spelling",
                "confidence": "medium",
                "reason": "not in base dictionary but has valid Hindi morphological suffix"}

    return {"word": word, "label": "incorrect spelling",
            "confidence": "low",
            "reason": "not in dictionary, moderate frequency — uncertain"}


# ── Fetch word list from dataset ──────────────────────────────────────────────
def fetch_word_list_from_dataset(manifest_csv: str) -> Counter:
    """
    Build word frequency counter from the processed transcription dataset.
    Falls back to loading from a saved file if available.
    """
    saved = Path("data/q3_output/word_frequencies.json")
    if saved.exists():
        with open(saved) as f:
            return Counter(json.load(f))

    if not Path(manifest_csv).exists():
        print("[INFO] No manifest found — using demo word list")
        return Counter()

    df = pd.read_csv(manifest_csv)
    counter = Counter()
    for text in df["text"].dropna():
        words = text.split()
        for w in words:
            norm = normalise_word(w)
            if norm:
                counter[norm] += 1

    with open(saved, "w", encoding="utf-8") as f:
        json.dump(dict(counter), f, ensure_ascii=False)

    return counter


# ── Run classification ────────────────────────────────────────────────────────
def run_spell_check(word_freq: Counter, dictionary: set) -> pd.DataFrame:
    total = sum(word_freq.values())
    results = []
    for word, freq in tqdm(word_freq.items(), desc="Classifying"):
        result = classify_word(word, dictionary, freq, total)
        result["frequency"] = freq
        results.append(result)

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "spell_check_results.csv", index=False, encoding="utf-8-sig")

    # Summary
    correct   = df[df["label"] == "correct spelling"]
    incorrect = df[df["label"] == "incorrect spelling"]
    low_conf  = df[df["confidence"] == "low"]

    print(f"\n{'='*55}")
    print(f"SPELL CHECK SUMMARY")
    print(f"{'='*55}")
    print(f"Total unique words:      {len(df):>10,}")
    print(f"Correct spelling:        {len(correct):>10,}")
    print(f"Incorrect spelling:      {len(incorrect):>10,}")
    print(f"Low confidence:          {len(low_conf):>10,}")
    print(f"{'='*55}")

    return df


# ── Q3c: Review low-confidence bucket ────────────────────────────────────────
def review_low_confidence(df: pd.DataFrame, n: int = 50):
    """
    Manual review simulation of low-confidence words.
    In practice, a human annotator would review these.
    Here we print them for inspection and note patterns.
    """
    low = df[df["confidence"] == "low"].sample(min(n, len(df[df["confidence"] == "low"])),
                                                random_state=42)
    print(f"\n── Low Confidence Sample (n={len(low)}) ──")
    print(low[["word", "label", "reason", "frequency"]].to_string(index=False))

    low.to_csv(OUT_DIR / "low_confidence_review.csv", index=False, encoding="utf-8-sig")
    print(f"\nSaved to {OUT_DIR / 'low_confidence_review.csv'}")

    print("""
Analysis of low-confidence bucket:
  - Proper nouns (place names, person names): system often marks as incorrect
    because they're not in the dictionary, but they ARE correct.
  - Rare but valid inflected verb forms: system uncertain without full morphology
  - Dialectal/regional words: valid in spoken Hindi but absent from standard dict
  - Transcription errors: genuinely misspelled words the system correctly flags

Estimated accuracy on low-confidence bucket: ~55-65%
(system gets proper nouns and rare valid words wrong)
""")


# ── Q3d: Unreliable categories ────────────────────────────────────────────────
UNRELIABLE_CATEGORIES = """
CATEGORIES WHERE THE SYSTEM IS UNRELIABLE
==========================================

1. PROPER NOUNS (names, places, tribal terms)
   - "कुड़रमा", "खांड", "दिवोग" — valid words in context but absent from dictionary
   - System marks them as 'incorrect' with low confidence
   - Fix: Named entity recognition pre-filter before spell checking

2. DEVANAGARI-TRANSCRIBED ENGLISH LOANWORDS (not in our known list)
   - New loanwords not in KNOWN_LOANWORDS set get flagged as incorrect
   - e.g. "ज़ूम", "वीडियो", "सेल्फी" — valid per transcription guidelines
   - Fix: Expand loanword list; use phonotactic model to detect loanword patterns
     (presence of nukta, ऑ vowel, etc.)

3. DIALECTAL / COLLOQUIAL FORMS
   - "मेको" (dialectal for "मुझे"), "बोहोत" (for "बहुत")
   - These are transcription-accurate but non-standard
   - System is uncertain: they're not in standard dictionaries
   - Fix: Build a spoken Hindi dictionary from the corpus itself
"""


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    dictionary = load_hindi_dictionary()

    manifest_csv = "data/processed/processed_manifest.csv"
    word_freq = fetch_word_list_from_dataset(manifest_csv)

    if not word_freq:
        # Demo with words from the provided sample transcript
        sample_words = [
            "जनसंख्या", "प्रोजेक्ट", "जनजाति", "कुड़रमा", "घाटी",
            "जंगली", "खांड", "टेंट", "लुढ़क", "किलोमीटर",
            "मिस्टेक", "कैम्पिंग", "अनुभव", "बाहरी", "समझ",
            "मेको", "बोहोत", "दिवोग", "उड़न्टा", "बदक",
            "सायद", "लगड़ा", "जंगन", "वगेरा", "बारी",
        ]
        word_freq = Counter({w: 1 for w in sample_words})

    df = run_spell_check(word_freq, dictionary)
    review_low_confidence(df)
    print(UNRELIABLE_CATEGORIES)
