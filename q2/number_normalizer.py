"""
Q2a — Hindi number word to digit normaliser.
Handles simple, compound, and large numbers.
Protects idiomatic phrases from conversion.
"""

import re

# Complete ONES table: 0-99
ONES = {
    "शून्य": 0,
    "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पाँच": 5, "पांच": 5,
    "छः": 6, "छह": 6, "छे": 6, "सात": 7, "आठ": 8, "नौ": 9,
    "दस": 10, "ग्यारह": 11, "बारह": 12, "तेरह": 13, "चौदह": 14,
    "पंद्रह": 15, "सोलह": 16, "सत्रह": 17, "अठारह": 18, "उन्नीस": 19,
    "बीस": 20, "इक्कीस": 21, "बाईस": 22, "तेईस": 23, "चौबीस": 24,
    "पच्चीस": 25, "छब्बीस": 26, "सत्ताईस": 27, "अट्ठाईस": 28, "उनतीस": 29,
    "तीस": 30, "इकतीस": 31, "बत्तीस": 32, "तैंतीस": 33, "चौंतीस": 34,
    "पैंतीस": 35, "छत्तीस": 36, "सैंतीस": 37, "अड़तीस": 38, "उनतालीस": 39,
    "चालीस": 40, "इकतालीस": 41, "बयालीस": 42, "तैंतालीस": 43, "चवालीस": 44,
    "पैंतालीस": 45, "छियालीस": 46, "सैंतालीस": 47, "अड़तालीस": 48, "उनचास": 49,
    "पचास": 50, "इक्यावन": 51, "बावन": 52, "तिरपन": 53, "चौवन": 54,
    "पचपन": 55, "छप्पन": 56, "सत्तावन": 57, "अट्ठावन": 58, "उनसठ": 59,
    "साठ": 60, "इकसठ": 61, "बासठ": 62, "तिरसठ": 63, "चौंसठ": 64,
    "पैंसठ": 65, "छियासठ": 66, "सड़सठ": 67, "अड़सठ": 68, "उनहत्तर": 69,
    "सत्तर": 70, "इकहत्तर": 71, "बहत्तर": 72, "तिहत्तर": 73, "चौहत्तर": 74,
    "पचहत्तर": 75, "छिहत्तर": 76, "सतहत्तर": 77, "अठहत्तर": 78, "उनासी": 79,
    "अस्सी": 80, "इक्यासी": 81, "बयासी": 82, "तिरासी": 83, "चौरासी": 84,
    "पचासी": 85, "छियासी": 86, "सत्तासी": 87, "अट्ठासी": 88, "नवासी": 89,
    "नब्बे": 90, "इक्यानवे": 91, "बानवे": 92, "तिरानवे": 93, "चौरानवे": 94,
    "पचानवे": 95, "छियानवे": 96, "सत्तानवे": 97, "अट्ठानवे": 98, "निन्यानवे": 99,
}

MULTIPLIERS = {
    "सौ": 100,
    "हज़ार": 1000, "हजार": 1000,
    "लाख": 100_000,
    "करोड़": 10_000_000,
}

# Idiomatic phrases — number conversion would be semantically wrong.
# IMPORTANT: only match hyphenated forms (दो-चार) as idioms.
# Space-separated forms (छः सात आठ) are literal enumerations and MUST be converted.
# Hyphen is the key marker that signals idiomatic/approximate usage in Hindi.
IDIOM_PATTERNS = [
    re.compile(r'दो-चार'),       # "a few things" — idiomatic
    re.compile(r'दो-तीन'),       # "two or three" — approximate
    re.compile(r'तीन-चार'),
    re.compile(r'चार-छः'),
    re.compile(r'पाँच-सात'),
    re.compile(r'सात-आठ'),
    re.compile(r'एक-दो'),
    re.compile(r'एक[-\s]न[-\s]एक'),  # "one or the other" — always idiomatic
    re.compile(r'दो-टूक'),            # "blunt/direct"
    re.compile(r'चार-चाँद'),          # "to glorify"
]

ALL_NUMBER_WORDS = set(ONES.keys()) | set(MULTIPLIERS.keys())


def parse_hindi_number(tokens: list) -> tuple:
    """
    Greedily parse Hindi number tokens into an integer.
    Returns (value, tokens_consumed) or (None, 0).

    Algorithm:
      A number span is: [ONES] [MULTIPLIER [ONES]]*
      Key rule: an ONES token is only consumed as part of the current number
      if it is either (a) the first token, or (b) immediately follows a MULTIPLIER.
      Two consecutive ONES tokens without a multiplier between them are
      independent numbers (e.g. "छः सात आठ" = 6, 7, 8 — not 21).

    Example: तीन सौ चौवन
      consume तीन(3), see सौ(100) -> current=300
      multiplier seen, so consume next ONES: चौवन(54) -> current=354
      end -> total=354  ✓

    Example: एक हज़ार पाँच सौ
      consume एक(1), see हज़ार(1000) -> result=1000, current=0
      multiplier seen, consume पाँच(5), see सौ(100) -> current=500
      end -> total=1500  ✓

    Example: छः सात आठ  (independent numbers)
      consume छः(6), next token सात is ONES with no preceding multiplier -> STOP
      returns (6, 1)  ✓
    """
    if not tokens or tokens[0] not in ALL_NUMBER_WORDS:
        return None, 0

    result = 0
    current = 0
    i = 0
    just_saw_multiplier = False  # track whether last token was a multiplier

    while i < len(tokens):
        tok = tokens[i]
        if tok in ONES:
            # Only consume this ONES token if it's the first token OR
            # we just processed a multiplier (compound number continuation)
            if i == 0 or just_saw_multiplier:
                current += ONES[tok]
                just_saw_multiplier = False
                i += 1
            else:
                break  # independent number — stop here
        elif tok in MULTIPLIERS:
            mult = MULTIPLIERS[tok]
            if current == 0:
                current = 1
            if mult >= 1000:
                result += current * mult
                current = 0
            else:
                current = current * mult
            just_saw_multiplier = True
            i += 1
        else:
            break

    total = result + current
    return (total if total > 0 else None), i


def normalise_numbers(text: str) -> str:
    """
    Convert Hindi number words to digits, skipping idiomatic phrases.

    Examples:
      "चौदह किताबें"              -> "14 किताबें"
      "तीन सौ चौवन रुपये"        -> "354 रुपये"
      "एक हज़ार पाँच सौ मीटर"   -> "1500 मीटर"
      "छः सात आठ किलोमीटर"      -> "6 7 8 किलोमीटर"  (independent numbers)
      "दो-चार बातें"              -> "दो-चार बातें"    (idiom, kept as-is)
    """
    # Step 1: protect idioms by replacing with placeholder tokens
    protected = {}
    protected_text = text
    for idx, pat in enumerate(IDIOM_PATTERNS):
        def protect(m, i=idx):
            key = f"__IDIOM{i}_{m.start()}__"
            protected[key] = m.group(0)
            return key
        protected_text = pat.sub(protect, protected_text)

    # Step 2: tokenise and greedily convert number spans
    words = protected_text.split()
    output = []
    i = 0
    while i < len(words):
        w = words[i]
        # Restore protected idiom
        if w.startswith("__IDIOM"):
            output.append(protected.get(w, w))
            i += 1
            continue
        # Try to parse a number starting here
        value, consumed = parse_hindi_number(words[i:])
        if value is not None and consumed > 0:
            output.append(str(value))
            i += consumed
        else:
            output.append(w)
            i += 1

    return " ".join(output)
