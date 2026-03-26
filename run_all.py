"""
Main runner — executes all four questions in sequence.
Q1 fine-tuning is skipped by default (takes hours); set RUN_FINETUNE=True to enable.
"""

import sys

RUN_FINETUNE = False   # set True to run full fine-tuning

print("\n" + "="*70)
print("JOSH TALKS — AI RESEARCHER INTERN ASSIGNMENT")
print("="*70)

# ── Q2: Number normalisation + English detection (no GPU needed) ──────────────
print("\n[Q2] Running ASR Cleanup Pipeline demos...")
from q2_cleanup_pipeline import demo_number_normalisation, demo_english_detection
demo_number_normalisation()
demo_english_detection()

# ── Q3: Spell checking ────────────────────────────────────────────────────────
print("\n[Q3] Running Spell Check on sample words...")
from q3_spell_check import load_hindi_dictionary, run_spell_check, review_low_confidence, UNRELIABLE_CATEGORIES
from collections import Counter

dictionary = load_hindi_dictionary()
# Demo with words from the provided transcript
sample_words = [
    "जनसंख्या", "प्रोजेक्ट", "जनजाति", "कुड़रमा", "घाटी",
    "जंगली", "खांड", "टेंट", "लुढ़क", "किलोमीटर",
    "मिस्टेक", "कैम्पिंग", "अनुभव", "बाहरी", "समझ",
    "मेको", "बोहोत", "दिवोग", "उड़न्टा", "बदक",
    "सायद", "लगड़ा", "जंगन", "वगेरा", "बारी",
    "अच्छा", "होता", "क्योंकि", "देखना", "लिखना",
    "गया", "थे", "हम", "वहाँ", "पहले",
]
word_freq = Counter({w: max(1, 10 - i) for i, w in enumerate(sample_words)})
df_spell = run_spell_check(word_freq, dictionary)
review_low_confidence(df_spell, n=10)
print(UNRELIABLE_CATEGORIES)

# ── Q4: Lattice-based WER ─────────────────────────────────────────────────────
print("\n[Q4] Running Lattice-based WER Evaluation...")
from q4_lattice_wer import run_evaluation
run_evaluation()

# ── Q1: Error taxonomy (no GPU needed) ───────────────────────────────────────
print("\n[Q1e/f] Printing Error Taxonomy and Fixes...")
from q1.error_analysis.taxonomy import print_taxonomy
from q1.error_analysis.fixes import FIX_1, FIX_3
print_taxonomy()
print(FIX_1)
print(FIX_3)

# ── Q1: Fine-tuning (GPU, optional) ──────────────────────────────────────────
if RUN_FINETUNE:
    print("\n[Q1b/c] Running Whisper fine-tuning (this will take several hours)...")
    import subprocess
    subprocess.run([sys.executable, "q1_finetune.py"])
else:
    print("\n[Q1b/c] Fine-tuning skipped (set RUN_FINETUNE=True in run_all.py to enable)")
    print("        Run manually: python q1_finetune.py")

print("\n" + "="*70)
print("All demos complete.")
print("="*70)
