"""
End-to-end verification script.
Tests Q2 number normalisation, Q2 English detection, Q3 classifier, Q4 alignment + lattice WER.
Uses the exact transcript segments provided in the assignment.
"""

import sys, re, unicodedata
from collections import Counter

PASS = 0
FAIL = 0

def check(label, got, expected):
    global PASS, FAIL
    if got == expected:
        print(f"  PASS  {label}")
        PASS += 1
    else:
        print(f"  FAIL  {label}")
        print(f"        expected: {expected!r}")
        print(f"        got:      {got!r}")
        FAIL += 1

# ─────────────────────────────────────────────────────────────────────────────
# Q2a — Number Normalisation
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Q2a: Number Normalisation ──")
sys.path.insert(0, ".")
from q2.number_normalizer import normalise_numbers, parse_hindi_number

check("simple: चौदह -> 14",
      normalise_numbers("उसने चौदह किताबें खरीदीं"),
      "उसने 14 किताबें खरीदीं")

check("compound: तीन सौ चौवन -> 354",
      normalise_numbers("तीन सौ चौवन रुपये दिए"),
      "354 रुपये दिए")

check("two-digit: पच्चीस -> 25",
      normalise_numbers("पच्चीस लोग आए थे"),
      "25 लोग आए थे")

check("large: एक हज़ार पाँच सौ -> 1500",
      normalise_numbers("एक हज़ार पाँच सौ मीटर दूर है"),
      "1500 मीटर दूर है")

check("sequence from transcript: छः सात -> 6 7",
      normalise_numbers("छः सात आठ किलोमीटर"),
      "6 7 8 किलोमीटर")

# Edge cases — idioms must be preserved
check("idiom: दो-चार kept as-is",
      normalise_numbers("दो-चार बातें करनी हैं"),
      "दो-चार बातें करनी हैं")

check("idiom: एक न एक kept as-is",
      normalise_numbers("एक न एक दिन आएगा"),
      "एक न एक दिन आएगा")

# From actual transcript: "छै सात में" (छै is variant of छः)
result_transcript = normalise_numbers("शाम मतलब छै सात में इतना")
print(f"  INFO  transcript 'छै सात में' -> {result_transcript!r}  (छै not in table, stays as word)")

# ─────────────────────────────────────────────────────────────────────────────
# Q2b — English Word Detection
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Q2b: English Word Detection ──")
from q2.english_detector import tag_english_words, get_english_words

check("Roman script tagged",
      tag_english_words("मेरा interview अच्छा गया"),
      "मेरा [EN]interview[/EN] अच्छा गया")

check("Devanagari loanword tagged",
      tag_english_words("हमारा प्रोजेक्ट भी था"),
      "हमारा [EN]प्रोजेक्ट[/EN] भी था")

check("multiple Roman words",
      tag_english_words("ये problem solve नहीं हो रहा"),
      "ये [EN]problem[/EN] [EN]solve[/EN] नहीं हो रहा")

# From actual transcript
transcript_text = "हम लोग टेंट वगेरा अगर कहीं भी कैम्पिंग करने जाते हैं"
en_words = get_english_words(transcript_text)
print(f"  INFO  transcript loanwords detected: {en_words}")
assert "टेंट" in en_words or "कैम्पिंग" in en_words, f"Expected loanwords not detected, got: {en_words}"
print(f"  PASS  transcript loanwords detected correctly")
PASS += 1

en2 = get_english_words("हम ने मिस्टेक किए कि हम लाइट नहीं ले गए थे")
print(f"  INFO  'मिस्टेक', 'लाइट' detected: {en2}")

# ─────────────────────────────────────────────────────────────────────────────
# Q3 — Spell Checker
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Q3: Spell Checker ──")
from q3.classifier import classify_word, normalise_word
from q3.dictionary_loader import load_hindi_dictionary

dictionary = load_hindi_dictionary()
total_words = 177000

# Known loanword — must be correct/high
r = classify_word("प्रोजेक्ट", dictionary, 50, total_words)
check("loanword प्रोजेक्ट -> correct/high",
      (r["label"], r["confidence"]),
      ("correct spelling", "high"))

# Roman characters — must be incorrect/high
r = classify_word("project", dictionary, 50, total_words)
check("Roman 'project' -> incorrect/high",
      (r["label"], r["confidence"]),
      ("incorrect spelling", "high"))

# Common Hindi word
r = classify_word("जनजाति", dictionary, 100, total_words)
print(f"  INFO  'जनजाति' -> {r['label']} ({r['confidence']}): {r['reason']}")

# Rare word not in dict — should be low confidence
r = classify_word("कुड़रमा", dictionary, 1, total_words)
check("rare OOV 'कुड़रमा' -> low confidence",
      r["confidence"], "low")

# Dialectal word from transcript
r = classify_word("मेको", dictionary, 3, total_words)
print(f"  INFO  dialectal 'मेको' -> {r['label']} ({r['confidence']}): {r['reason']}")

# Obvious misspelling (double matra test)
bad_word = "कि\u093Eत\u093E"  # किाता — double matra
r = classify_word(bad_word, dictionary, 5, total_words)
check("double matra -> incorrect/high",
      (r["label"], r["confidence"]),
      ("incorrect spelling", "high"))

# ─────────────────────────────────────────────────────────────────────────────
# Q4 — Alignment + Lattice WER
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Q4: Alignment ──")
from q4.alignment import edit_distance_alignment, compute_standard_wer

ref = ["उसने", "चौदह", "किताबें", "खरीदीं"]
hyp = ["उसने", "14",    "किताबें", "खरीदी"]
alignment = edit_distance_alignment(ref, hyp)
print(f"  INFO  alignment: {alignment}")

# Should have 2 substitutions, 0 deletions, 0 insertions
subs = sum(1 for r,h in alignment if r and h and r != h)
check("alignment: 2 substitutions", subs, 2)

std_wer = compute_standard_wer(ref, hyp)
check("standard WER = 50.0%", std_wer, 50.0)

# Deletion test
ref2 = ["तो", "हम", "वहाँ", "गए", "थे"]
hyp2 = ["हम", "वहाँ", "गए", "थे"]
dels = sum(1 for r,h in edit_distance_alignment(ref2, hyp2) if r and not h)
check("deletion: 1 deletion (तो)", dels, 1)

# Insertion test
ref3 = ["बहुत", "अजीब", "सा", "आवाज"]
hyp3 = ["बहुत", "ही", "अजीब", "सा", "आवाज"]
ins = sum(1 for r,h in edit_distance_alignment(ref3, hyp3) if not r and h)
check("insertion: 1 insertion (ही)", ins, 1)

print("\n── Q4: Lattice WER ──")
from q4.lattice import build_lattice, compute_lattice_wer, LatticeBin

# Test: "14" should match bin containing "चौदह" via KNOWN_ALTERNATIVES
ref_seg = ["उसने", "चौदह", "किताबें", "खरीदीं"]
model_outputs = {
    "m1": ["उसने", "14",     "किताबें", "खरीदीं"],
    "m2": ["उसने", "14",     "किताबें", "खरीदी"],
    "m3": ["उसने", "चौदह",  "किताबें", "खरीदीं"],
    "m4": ["उसने", "14",     "किताबें", "खरीदीं"],
    "m5": ["उसने", "चौदह",  "किताबें", "खरीदी"],
}
lattice = build_lattice(ref_seg, model_outputs)
print(f"  INFO  lattice bins: {[list(b.tokens) for b in lattice]}")

# m1 hypothesis: उसने 14 किताबें खरीदीं
# "14" should match bin[1] because 3/5 models say "14" -> model agreement override
lat_result = compute_lattice_wer(lattice, ["उसने", "14", "किताबें", "खरीदीं"], ref_seg)
std_result = compute_standard_wer(ref_seg, ["उसने", "14", "किताबें", "खरीदीं"])
print(f"  INFO  m1 standard WER: {std_result}%  lattice WER: {lat_result['wer']}%")
check("lattice WER <= standard WER for unfairly penalised model",
      lat_result["wer"] <= std_result, True)

# m3 hypothesis: exact match — lattice WER should equal standard WER
lat_exact = compute_lattice_wer(lattice, ref_seg, ref_seg)
std_exact  = compute_standard_wer(ref_seg, ref_seg)
check("exact match: lattice WER == standard WER == 0.0",
      (lat_exact["wer"], std_exact), (0.0, 0.0))

# Test with actual transcript segment from assignment
print("\n── Q4: Full segment test (from assignment transcript) ──")
ref_full = "अब काफी अच्छा होता है क्योंकि उनकी जनसंख्या बहुत कम दी जा रही है".split()
hyp_full_a = "अब काफी अच्छा होता है क्योंकि उनकी जनसंख्या बहुत कम दी जा रही है".split()
hyp_full_b = "अब काफी अच्छा होता क्योंकि उनकी जनसंख्या बहुत कम दी जा रही है".split()  # deletion

models_full = {"mA": hyp_full_a, "mB": hyp_full_b,
               "mC": hyp_full_a, "mD": hyp_full_a, "mE": hyp_full_b}
lattice_full = build_lattice(ref_full, models_full)
wer_a = compute_lattice_wer(lattice_full, hyp_full_a, ref_full)
wer_b = compute_lattice_wer(lattice_full, hyp_full_b, ref_full)
print(f"  INFO  mA (exact match) lattice WER: {wer_a['wer']}%")
print(f"  INFO  mB (1 deletion)  lattice WER: {wer_b['wer']}%")
check("mA exact match -> 0% lattice WER", wer_a["wer"], 0.0)
check("mB deletion -> WER > 0%", wer_b["wer"] > 0, True)

# ─────────────────────────────────────────────────────────────────────────────
# Q1 modules — import check
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Q1: Module import checks ──")
try:
    from q1.preprocess.url_builder import build_url
    url = build_url("967179", "825780", "transcription")
    check("URL builder",
          url,
          "https://storage.googleapis.com/upload_goai/967179/825780_transcription.json")
except Exception as e:
    print(f"  FAIL  url_builder: {e}"); FAIL += 1

try:
    from q1.preprocess.text_normalizer import normalize_text
    check("text normalizer strips danda",
          normalize_text("नमस्ते। कैसे हो?"),
          "नमस्ते कैसे हो")
except Exception as e:
    print(f"  FAIL  text_normalizer: {e}"); FAIL += 1

try:
    from q1.error_analysis.taxonomy import TAXONOMY, print_taxonomy
    check("taxonomy has 7 categories", len(TAXONOMY), 7)
except Exception as e:
    print(f"  FAIL  taxonomy: {e}"); FAIL += 1

try:
    from q1.error_analysis.fixes import normalise_script
    check("script fix: project -> प्रोजेक्ट",
          normalise_script("हमारा project था"),
          "हमारा प्रोजेक्ट था")
except Exception as e:
    print(f"  FAIL  fixes: {e}"); FAIL += 1

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Results: {PASS} passed, {FAIL} failed")
print(f"{'='*50}")
sys.exit(0 if FAIL == 0 else 1)
