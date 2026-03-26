"""
Q1f/g — Top-3 fixes for most frequent error types.
Implements Fix #2 (script normalisation) with before/after WER comparison.
"""

import re
import pandas as pd
from jiwer import wer
from pathlib import Path

# ── Fix #1: Diacritic corrector (proposed, not implemented here) ──────────────
FIX_1 = """
FIX #1 — DIACRITIC / MATRA ERRORS
  Approach: Train a character-level seq2seq corrector on (noisy -> clean) pairs.
  Generate training data by artificially dropping anusvara/chandrabindu from
  the existing transcription corpus. Apply as a post-processing step after
  Whisper decoding — no retraining of the main model needed.
  Why better than more data: Diacritic errors are systematic, not random.
  A small corrector (few MB) fixes them reliably and cheaply.
"""

# ── Fix #2: Script normalisation (implemented) ───────────────────────────────
ROMAN_TO_DEVANAGARI = {
    "project": "प्रोजेक्ट", "tent": "टेंट", "area": "एरिया",
    "interview": "इंटरव्यू", "camp": "कैम्प", "camping": "कैम्पिंग",
    "light": "लाइट", "amazon": "अमेज़न", "guard": "गार्ड",
    "problem": "प्रॉब्लम", "solve": "सॉल्व", "job": "जॉब",
    "enter": "एंटर", "road": "रोड",
}

_ROMAN_TOKEN = re.compile(r'\b[a-zA-Z]+\b')


def normalise_script(text: str) -> str:
    """Replace Roman tokens with Devanagari equivalents where known."""
    return _ROMAN_TOKEN.sub(
        lambda m: ROMAN_TO_DEVANAGARI.get(m.group(0).lower(), m.group(0)),
        text
    )


def apply_script_fix(df: pd.DataFrame, out_dir: Path) -> None:
    """Apply script normalisation and report before/after WER."""
    cs_mask = df["hypothesis"].str.contains(_ROMAN_TOKEN, regex=True, na=False)
    subset  = df[cs_mask].copy()

    if subset.empty:
        print("No code-switch errors found.")
        return

    subset["hypothesis_fixed"] = subset["hypothesis"].apply(normalise_script)
    before = wer(list(subset["reference"]), list(subset["hypothesis"])) * 100
    after  = wer(list(subset["reference"]), list(subset["hypothesis_fixed"])) * 100

    print(f"\n── Fix #2: Script Normalisation ──")
    print(f"  Subset:      {len(subset)} utterances")
    print(f"  WER before:  {before:.2f}%")
    print(f"  WER after:   {after:.2f}%")
    print(f"  Improvement: {before - after:+.2f}%")

    print("\n  Sample before/after:")
    for _, row in subset.head(5).iterrows():
        if row["hypothesis"] != row["hypothesis_fixed"]:
            print(f"    REF:    {row['reference']}")
            print(f"    BEFORE: {row['hypothesis']}")
            print(f"    AFTER:  {row['hypothesis_fixed']}")
            print()

    out_dir.mkdir(parents=True, exist_ok=True)
    subset[["reference", "hypothesis", "hypothesis_fixed"]].to_csv(
        out_dir / "fix2_script_normalisation.csv", index=False
    )


# ── Fix #3: OOV vocabulary augmentation (proposed) ───────────────────────────
FIX_3 = """
FIX #3 — OOV / RARE WORD ERRORS
  Approach: Augment the Whisper tokenizer with domain-specific terms
  (tribal names, place names, loanwords). Initialise new token embeddings
  from phonetically similar existing tokens. Use a Hindi n-gram LM for
  shallow fusion at decode time to bias toward known vocabulary.
  Why better than more data: Rare words may never appear enough times
  in training to be learned. Explicit vocabulary injection is more efficient.
"""
