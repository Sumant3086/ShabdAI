"""
Q3 — Hindi dictionary loader.
Fetches IndicNLP Hindi wordlist from GitHub; falls back to built-in set.
"""

import requests

INDIC_LEXICON_URL = (
    "https://raw.githubusercontent.com/anoopkunchukuttan/"
    "indic_nlp_resources/master/lexicon/hi.txt"
)

FALLBACK_WORDS = {
    "है", "हैं", "था", "थे", "थी", "हो", "होना", "करना", "जाना",
    "आना", "देना", "लेना", "बोलना", "कहना", "सुनना", "देखना",
    "और", "या", "लेकिन", "क्योंकि", "तो", "भी", "ही", "न", "नहीं",
    "मैं", "हम", "तुम", "आप", "वो", "वह", "यह", "ये", "वे",
    "का", "की", "के", "को", "से", "में", "पर", "पे", "ने",
    "एक", "दो", "तीन", "चार", "पाँच", "दस", "सौ", "हज़ार",
    "बहुत", "थोड़ा", "ज़्यादा", "कम", "अच्छा", "बुरा", "बड़ा", "छोटा",
    "घर", "काम", "बात", "दिन", "रात", "समय", "लोग", "जगह",
    "पता", "मतलब", "अब", "फिर", "जब", "तब", "यहाँ", "वहाँ",
    "जनजाति", "जंगल", "घाटी", "अनुभव", "समझ", "भाषा",
}


def load_hindi_dictionary() -> set:
    """
    Load Hindi word dictionary.
    Tries IndicNLP lexicon first; falls back to built-in set.
    """
    try:
        r = requests.get(INDIC_LEXICON_URL, timeout=15)
        if r.status_code == 200:
            words = set(r.text.strip().split('\n'))
            print(f"[INFO] Loaded {len(words):,} words from IndicNLP lexicon")
            return words
    except Exception as e:
        print(f"[WARN] Could not fetch dictionary: {e}")

    print(f"[INFO] Using fallback dictionary ({len(FALLBACK_WORDS)} words)")
    return FALLBACK_WORDS
