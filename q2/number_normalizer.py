"""
Q2a — Hindi number word to digit normaliser.
Handles simple, compound, and large numbers.
Protects idiomatic phrases from conversion.
"""

import re

ONES = {
    "शून्य": 0, "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पाँच": 5, "पांच": 5,
    "छः": 6, "छह": 6, "सात": 7, "आठ": 8, "नौ": 9, "दस": 10,
    "ग्यारह": 11, "बारह": 12, "तेरह": 13, "चौदह": 14, "पंद्रह": 15,
    "सोलह": 16, "सत्रह": 17, "अठारह": 18, "उन्नीस": 19, "बीस": 20,
    "इक्कीस": 21, "बाईस": 22, "तेईस": 23, "चौबीस": 24, "पच्चीस": 25,
    "छब्बीस": 26, "सत्ताईस": 27, "अट्ठाईस": 28, "उनतीस": 29, "तीस": 30,
    "इकतीस": 31, "बत्तीस": 32, "तैंतीस": 33, "चौंतीस": 34, "पैंतीस": 35,
    "छत्तीस": 36, "सैंतीस": 37, "अड़तीस": 38, "उनतालीस": 39, "चालीस": 40,
    "पचास": 50, "साठ": 60, "सत्तर": 70, "अस्सी": 80, "नब्बे": 90,
}

MULTIPLIERS = {
    "सौ": 100, "हज़ार": 1000, "हजार": 1000,
    "लाख": 100_000, "करोड़": 10_000_000,
}

# Idiomatic phrases — number conversion would be semantically wrong
IDIOM_PATTERNS = [
    re.compile(r'दो[-\s]चार'),    # "a few"
    re.compile(r'दो[-\s]तीन'),    # "two or three"
    re.compile(r'तीन[-\s]चार'),
    re.compile(r'चार[-\s]छः'),
    re.compile(r'पाँच[-\s]सात'),
    re.compile(r'सात[-\s]आठ'),
    re.compile(r'एक[-\s]दो'),
    re.compile(r'एक[-\s]न[-\s]एक'),   # "one or the other"
    re.compile(r'दो[-\s]टूक'),         # "blunt/direct"
    re.compile(r'चार[-\s]चाँद'),       # "to glorify"
]


def parse_hindi_number(tokens: list) -> tuple:
    """
    Greedily parse Hindi number tokens into an integer.
    Returns (value, tokens_consumed) or (None, 0).
    """
    if not tokens or (tokens[0] not in ONES and tokens[0] not in MULTIPLIERS):
        return None, 0

    result = current = 0
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in ONES:
            current += ONES[tok]; i += 1
        elif tok in MULTIPLIERS:
            mult = MULTIPLIERS[tok]
            if current == 0:
                current = 1
            if mult >= 1000:
                result += current * mult; current = 0
            else:
                current *= mult
            i += 1
        else:
            break

    total = result + current
    return (total if total > 0 else None), i


def normalise_numbers(text: str) -> str:
    """
    Convert Hindi number words to digits, skipping idiomatic phrases.

    Examples:
      "चौदह किताबें"          -> "14 किताबें"
      "तीन सौ चौवन रुपये"    -> "354 रुपये"
      "एक हज़ार पाँच सौ"     -> "1500"
      "दो-चार बातें"          -> "दो-चार बातें"  (idiom, kept as-is)
    """
    # Protect idioms
    protected = {}
    protected_text = text
    for i, pat in enumerate(IDIOM_PATTERNS):
        def protect(m, idx=i):
            key = f"__IDIOM{idx}_{m.start()}__"
            protected[key] = m.group(0)
            return key
        protected_text = pat.sub(protect, protected_text)

    words = protected_text.split()
    output = []
    i = 0
    while i < len(words):
        if words[i].startswith("__IDIOM"):
            output.append(protected.get(words[i], words[i])); i += 1; continue
        value, consumed = parse_hindi_number(words[i:])
        if value is not None and consumed > 0:
            output.append(str(value)); i += consumed
        else:
            output.append(words[i]); i += 1

    return " ".join(output)
