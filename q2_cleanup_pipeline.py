"""
Q2 — ASR Cleanup Pipeline: Number Normalisation + English Word Detection

Runs pretrained Whisper-small on the dataset to generate raw ASR output,
then applies two cleanup operations.
"""

import re
import json
import torch
import requests
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from pathlib import Path
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID  = "openai/whisper-small"
OUT_DIR   = Path("data/q2_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# PART A — NUMBER NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────

# Hindi number word tables
ONES = {
    "शून्य": 0, "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पाँच": 5, "पांच": 5,
    "छः": 6, "छह": 6, "सात": 7, "आठ": 8, "नौ": 9, "दस": 10,
    "ग्यारह": 11, "बारह": 12, "तेरह": 13, "चौदह": 14, "पंद्रह": 15,
    "सोलह": 16, "सत्रह": 17, "अठारह": 18, "उन्नीस": 19, "बीस": 20,
    "इक्कीस": 21, "बाईस": 22, "तेईस": 23, "चौबीस": 24, "पच्चीस": 25,
    "छब्बीस": 26, "सत्ताईस": 27, "अट्ठाईस": 28, "उनतीस": 29, "तीस": 30,
    "इकतीस": 31, "बत्तीस": 32, "तैंतीस": 33, "चौंतीस": 34, "पैंतीस": 35,
    "छत्तीस": 36, "सैंतीस": 37, "अड़तीस": 38, "उनतालीस": 39, "चालीस": 40,
    "इकतालीस": 41, "बयालीस": 42, "तैंतालीस": 43, "चवालीस": 44, "पैंतालीस": 45,
    "छियालीस": 46, "सैंतालीस": 47, "अड़तालीस": 48, "उनचास": 49, "पचास": 50,
    "इक्यावन": 51, "बावन": 52, "तिरपन": 53, "चौवन": 54, "पचपन": 55,
    "छप्पन": 56, "सत्तावन": 57, "अट्ठावन": 58, "उनसठ": 59, "साठ": 60,
    "इकसठ": 61, "बासठ": 62, "तिरसठ": 63, "चौंसठ": 64, "पैंसठ": 65,
    "छियासठ": 66, "सड़सठ": 67, "अड़सठ": 68, "उनहत्तर": 69, "सत्तर": 70,
    "इकहत्तर": 71, "बहत्तर": 72, "तिहत्तर": 73, "चौहत्तर": 74, "पचहत्तर": 75,
    "छिहत्तर": 76, "सतहत्तर": 77, "अठहत्तर": 78, "उनासी": 79, "अस्सी": 80,
    "इक्यासी": 81, "बयासी": 82, "तिरासी": 83, "चौरासी": 84, "पचासी": 85,
    "छियासी": 86, "सत्तासी": 87, "अट्ठासी": 88, "नवासी": 89, "नब्बे": 90,
    "इक्यानवे": 91, "बानवे": 92, "तिरानवे": 93, "चौरानवे": 94, "पचानवे": 95,
    "छियानवे": 96, "सत्तानवे": 97, "अट्ठानवे": 98, "निन्यानवे": 99,
}

MULTIPLIERS = {
    "सौ": 100,
    "हज़ार": 1000, "हजार": 1000,
    "लाख": 100_000,
    "करोड़": 10_000_000,
}

# Idiomatic phrases where number conversion would be wrong
IDIOM_PATTERNS = [
    re.compile(r'दो[-\s]चार'),          # "दो-चार बातें" = a few things
    re.compile(r'चार[-\s]छः'),
    re.compile(r'दो[-\s]तीन'),
    re.compile(r'तीन[-\s]चार'),
    re.compile(r'पाँच[-\s]सात'),
    re.compile(r'सात[-\s]आठ'),
    re.compile(r'आठ[-\s]दस'),
    re.compile(r'एक[-\s]दो'),
    re.compile(r'एक[-\s]न[-\s]एक'),     # "एक न एक" = one or the other
    re.compile(r'दो[-\s]टूक'),          # "दो-टूक" = blunt/direct
    re.compile(r'चार[-\s]चाँद'),        # "चार चाँद लगाना" = to glorify
]


def is_idiomatic(span: str) -> bool:
    """Return True if the span is part of a known idiom."""
    for pat in IDIOM_PATTERNS:
        if pat.search(span):
            return True
    return False


def parse_hindi_number(tokens: list[str]) -> tuple[int | None, int]:
    """
    Greedily parse a sequence of Hindi number tokens into an integer.
    Returns (value, tokens_consumed) or (None, 0) if no number found.

    Handles:
      - Simple: ["दस"] -> 10
      - Compound: ["तीन", "सौ", "चौवन"] -> 354
      - Large: ["एक", "हज़ार", "पाँच", "सौ"] -> 1500
    """
    if not tokens or tokens[0] not in ONES and tokens[0] not in MULTIPLIERS:
        return None, 0

    result = 0
    current = 0
    i = 0

    while i < len(tokens):
        tok = tokens[i]
        if tok in ONES:
            current += ONES[tok]
            i += 1
        elif tok in MULTIPLIERS:
            mult = MULTIPLIERS[tok]
            if current == 0:
                current = 1
            if mult >= 1000:
                result += current * mult
                current = 0
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

    Edge case handling:
    - "दो-चार बातें" -> kept as-is (idiom)
    - "एक हज़ार" -> "1000"
    - "तीन सौ चौवन" -> "354"
    - "पच्चीस" -> "25"
    """
    # Protect idioms by temporarily replacing them
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
        # Skip protected idiom tokens
        if words[i].startswith("__IDIOM"):
            output.append(protected.get(words[i], words[i]))
            i += 1
            continue

        value, consumed = parse_hindi_number(words[i:])
        if value is not None and consumed > 0:
            output.append(str(value))
            i += consumed
        else:
            output.append(words[i])
            i += 1

    return " ".join(output)


# ── Before/After examples ─────────────────────────────────────────────────────
NUMBER_EXAMPLES = [
    # (input, expected_output, note)
    ("उसने चौदह किताबें खरीदीं",         "उसने 14 किताबें खरीदीं",       "simple"),
    ("तीन सौ चौवन रुपये दिए",            "354 रुपये दिए",                 "compound"),
    ("पच्चीस लोग आए थे",                 "25 लोग आए थे",                  "two-digit"),
    ("एक हज़ार पाँच सौ मीटर दूर है",     "1500 मीटर दूर है",              "large compound"),
    ("छः सात आठ किलोमीटर में नौ बजे",   "6 7 8 किलोमीटर में 9 बजे",     "sequence"),
    # Edge cases
    ("दो-चार बातें करनी हैं",            "दो-चार बातें करनी हैं",         "IDIOM: kept as-is"),
    ("एक न एक दिन आएगा",                "एक न एक दिन आएगा",              "IDIOM: kept as-is"),
    ("चार चाँद लगा दिए",                 "चार चाँद लगा दिए",              "IDIOM: kept as-is"),
]


def demo_number_normalisation():
    print("\n── Number Normalisation Examples ──")
    print(f"{'INPUT':<45} {'OUTPUT':<45} {'NOTE'}")
    print("-"*120)
    for inp, expected, note in NUMBER_EXAMPLES:
        result = normalise_numbers(inp)
        status = "OK" if result == expected else "DIFF"
        print(f"{inp:<45} {result:<45} [{note}] {status}")


# ─────────────────────────────────────────────────────────────────────────────
# PART B — ENGLISH WORD DETECTION
# ─────────────────────────────────────────────────────────────────────────────

# Common English loanwords that appear in Devanagari in Hindi conversation
# (per transcription guidelines: English spoken words -> Devanagari)
DEVANAGARI_ENGLISH_LOANWORDS = {
    # Technology / work
    "प्रोजेक्ट", "इंटरव्यू", "जॉब", "प्रॉब्लम", "सॉल्व", "कंप्यूटर",
    "मोबाइल", "फोन", "इंटरनेट", "ऑनलाइन", "ऑफलाइन", "वेबसाइट",
    # Nature / outdoor
    "टेंट", "कैम्प", "कैम्पिंग", "एरिया", "लाइट", "गार्ड",
    # Common loanwords
    "बस", "ट्रेन", "कार", "बाइक", "स्कूल", "कॉलेज", "क्लास",
    "टीचर", "डॉक्टर", "नर्स", "हॉस्पिटल", "मार्केट", "शॉप",
    "होटल", "रेस्टोरेंट", "पार्क", "रोड", "स्टेशन",
    # Amazon / jungle context
    "अमेज़न", "जंगल",
}

# Roman script detection
_ROMAN_WORD = re.compile(r'\b[a-zA-Z]{2,}\b')

# Devanagari Unicode range
_DEVANAGARI = re.compile(r'[\u0900-\u097F]')


def is_likely_english_devanagari(word: str) -> bool:
    """
    Heuristic: is this Devanagari word actually an English loanword?
    Checks against known loanword list.
    """
    return word in DEVANAGARI_ENGLISH_LOANWORDS


def tag_english_words(text: str) -> str:
    """
    Tag English words in a Hindi transcript.
    Handles two cases:
    1. Roman script words (model output Roman instead of Devanagari)
    2. Known English loanwords written in Devanagari

    Output format: [EN]word[/EN]

    Example:
      Input:  "मेरा interview अच्छा गया"
      Output: "मेरा [EN]interview[/EN] अच्छा गया"
    """
    # Tag Roman-script tokens
    def tag_roman(m):
        return f"[EN]{m.group(0)}[/EN]"
    tagged = _ROMAN_WORD.sub(tag_roman, text)

    # Tag known Devanagari loanwords
    words = tagged.split()
    result = []
    for word in words:
        # Skip already-tagged tokens
        if "[EN]" in word or "[/EN]" in word:
            result.append(word)
            continue
        # Strip punctuation for lookup
        clean = re.sub(r'[^\u0900-\u097F]', '', word)
        if clean and is_likely_english_devanagari(clean):
            result.append(f"[EN]{word}[/EN]")
        else:
            result.append(word)

    return " ".join(result)


# ── English detection examples ────────────────────────────────────────────────
ENGLISH_DETECTION_EXAMPLES = [
    "मेरा इंटरव्यू बहुत अच्छा गया और मुझे जॉब मिल गई",
    "हमारा प्रोजेक्ट भी था कि जो जनजाति पाई जाती है",
    "हम टेंट गड़ा और रहे",
    "मेरा interview अच्छा गया",                          # Roman script
    "ये problem solve नहीं हो रहा",                      # Mixed Roman
    "अमेज़न का जंगल होता है",
    "रोड पे होता है न रोड का जो एरिया",
]


def demo_english_detection():
    print("\n── English Word Detection Examples ──")
    for text in ENGLISH_DETECTION_EXAMPLES:
        tagged = tag_english_words(text)
        print(f"  IN:  {text}")
        print(f"  OUT: {tagged}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Generate raw ASR transcripts using pretrained Whisper-small
# ─────────────────────────────────────────────────────────────────────────────

def generate_raw_asr(audio_paths: list, reference_texts: list) -> pd.DataFrame:
    """
    Run pretrained Whisper-small on audio files to get raw ASR output.
    Pairs with human reference transcriptions.
    """
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    records = []
    for audio_path, ref in tqdm(zip(audio_paths, reference_texts), desc="ASR"):
        try:
            wav, sr = librosa.load(audio_path, sr=16_000, mono=True)
            inputs = processor(wav, sampling_rate=16_000, return_tensors="pt").to(device)
            with torch.no_grad():
                ids = model.generate(
                    inputs.input_features,
                    language="hi",
                    task="transcribe",
                )
            raw_asr = processor.batch_decode(ids, skip_special_tokens=True)[0]
            records.append({
                "audio_path":    audio_path,
                "reference":     ref,
                "raw_asr":       raw_asr,
                "num_norm":      normalise_numbers(raw_asr),
                "en_tagged":     tag_english_words(raw_asr),
                "both_ops":      tag_english_words(normalise_numbers(raw_asr)),
            })
        except Exception as e:
            print(f"  [WARN] {audio_path}: {e}")

    df = pd.DataFrame(records)
    df.to_csv(OUT_DIR / "asr_cleanup_output.csv", index=False)
    return df


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo_number_normalisation()
    demo_english_detection()

    # If processed audio exists, run full pipeline
    audio_dir = Path("data/audio")
    manifest  = Path("data/processed/processed_manifest.csv")
    if manifest.exists():
        df_manifest = pd.read_csv(manifest)
        audio_paths = [str(audio_dir / f"{r.user_id}_{r.recording_id}.wav")
                       for _, r in df_manifest.iterrows()]
        refs = df_manifest["text"].tolist()
        result_df = generate_raw_asr(audio_paths, refs)
        print(f"\nSaved cleanup output to {OUT_DIR / 'asr_cleanup_output.csv'}")
    else:
        print("\n[INFO] Run q1_preprocess.py first to generate audio files.")
