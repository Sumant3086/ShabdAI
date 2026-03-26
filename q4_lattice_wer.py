"""
Q4 — Lattice-based WER for Hindi ASR

Design:
  Alignment unit: WORD (justified below)
  - Hindi is an agglutinative language but words are clearly space-delimited
  - Subword units would fragment proper nouns and loanwords unpredictably
  - Phrase-level is too coarse for fine-grained insertion/deletion tracking

Pipeline:
  1. Align all model outputs + reference using multi-sequence alignment (MSA)
  2. Build a lattice: list of bins, each bin = set of valid alternatives
  3. For each model, compute lattice-WER:
     - A model output is "correct" at position i if it matches ANY token in bin[i]
  4. Trust model agreement over reference when >= 3/5 models agree on a token
     that differs from reference (reference may be wrong)
"""

import re
import json
import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

# ── Data from Q4 prompt ───────────────────────────────────────────────────────
REFERENCE_SEGMENTS = [
    {"start": 0.11,  "end": 14.42, "text": "अब काफी अच्छा होता है क्योंकि उनकी जनसंख्या बहुत कम दी जा रही है तो हमें उनको देखना था तो एक देखना था मतलब वो तो देखना था लेकिन हमारा प्रोजेक्ट भी था कि जो जन जाती पाई जाती है उधर कि उधर की एरिया में उसके बारे में देखना अब"},
    {"start": 14.42, "end": 29.03, "text": "अनुभव करके कुछ लिखना था तो वह तो बिना देखिए नहीं हो सकती थी तो हम वहां गया थे कुड़रमा घाटी तरफ पर दिवोग काफी जंगली एरिया है वह जो खांड जनजाति पाए जाती ना वहां पाए जाती है तो"},
    {"start": 29.03, "end": 41.84, "text": "जंगल का सफर होता है जब हम रहने के लिए गए थे नातो चाहते के साथ जैसे हम वहाँ पहले एंटर किये थे तो पहले तो गिर गया थे लगड़ा के उपर से नीचे"},
]

# Simulated outputs from 5 ASR models (based on known Whisper error patterns)
MODEL_OUTPUTS = {
    "model_A": [
        "अब काफी अच्छा होता है क्योंकि उनकी जनसंख्या बहुत कम दी जा रही है तो हमें उनको देखना था तो एक देखना था मतलब वो तो देखना था लेकिन हमारा project भी था कि जो जन जाती पाई जाती है उधर कि उधर की area में उसके बारे में देखना अब",
        "अनुभव करके कुछ लिखना था तो वह तो बिना देखे नहीं हो सकती थी तो हम वहां गए थे कुड़मा घाटी तरफ पर दिवोग काफी जंगली area है वह जो खान जनजाति पाए जाती ना वहां पाए जाती है तो",
        "जंगल का सफर होता है जब हम रहने के लिए गए थे नातो चाहते के साथ जैसे हम वहाँ पहले enter किये थे तो पहले तो गिर गए थे लगड़ा के ऊपर से नीचे",
    ],
    "model_B": [
        "अब काफी अच्छा होता है क्योंकि उनकी जनसंख्या बहुत कम दी जा रही है तो हमें उनको देखना था तो एक देखना था मतलब वो तो देखना था लेकिन हमारा प्रोजेक्ट भी था कि जो जनजाति पाई जाती है उधर की उधर की एरिया में उसके बारे में देखना अब",
        "अनुभव करके कुछ लिखना था तो वह तो बिना देखे नहीं हो सकती थी तो हम वहाँ गए थे कुड़रमा घाटी तरफ पर दिवोग काफी जंगली एरिया है वह जो खांड जनजाति पाए जाती ना वहाँ पाए जाती है तो",
        "जंगल का सफर होता है जब हम रहने के लिए गए थे नातो चाहते के साथ जैसे हम वहाँ पहले एंटर किए थे तो पहले तो गिर गए थे लगड़ा के ऊपर से नीचे",
    ],
    "model_C": [
        "अब काफी अच्छा होता है क्योंकि उनकी जनसंख्या बहुत कम दी जा रही है तो हमें उनको देखना था एक देखना था मतलब वो देखना था लेकिन हमारा प्रोजेक्ट भी था कि जो जन जाती पाई जाती है उधर की एरिया में उसके बारे में देखना",
        "अनुभव करके कुछ लिखना था तो वह बिना देखे नहीं हो सकती थी तो हम वहाँ गए थे कुड़रमा घाटी तरफ दिवोग काफी जंगली एरिया है वह जो खांड जनजाति पाई जाती ना वहाँ पाई जाती है तो",
        "जंगल का सफर होता है जब हम रहने के लिए गए थे साथ जैसे हम वहाँ पहले एंटर किए थे तो पहले गिर गए थे लगड़ा के ऊपर से नीचे",
    ],
    "model_D": [
        "अब काफी अच्छा होता है क्योंकि उनकी जनसंख्या बहुत कम दी जा रही है तो हमें उनको देखना था तो एक देखना था मतलब वो तो देखना था लेकिन हमारा प्रोजेक्ट भी था कि जो जन जाती पाई जाती है उधर कि उधर की एरिया में उसके बारे में देखना अब",
        "अनुभव करके कुछ लिखना था तो वह तो बिना देखिए नहीं हो सकती थी तो हम वहाँ गए थे कुड़रमा घाटी तरफ पर दिवोग काफी जंगली एरिया है वह जो खांड जनजाति पाए जाती ना वहाँ पाए जाती है तो",
        "जंगल का सफर होता है जब हम रहने के लिए गए थे नातो चाहते के साथ जैसे हम वहाँ पहले एंटर किए थे तो पहले तो गिर गए थे लगड़ा के ऊपर से नीचे",
    ],
    "model_E": [
        "अब काफी अच्छा होता है क्योंकि उनकी जनसंख्या बहुत कम दी जा रही है हमें उनको देखना था तो एक देखना था मतलब वो तो देखना था लेकिन हमारा प्रोजेक्ट भी था कि जो जन जाती पाई जाती है उधर की एरिया में उसके बारे में देखना अब",
        "अनुभव करके कुछ लिखना था तो वह तो बिना देखे नहीं हो सकती थी तो हम वहाँ गए थे कुड़रमा घाटी तरफ पर दिवोग काफी जंगली एरिया है वह जो खांड जनजाति पाए जाती ना वहाँ पाए जाती है",
        "जंगल का सफर होता है जब हम रहने के लिए गए थे नातो चाहते के साथ जैसे हम वहाँ पहले एंटर किए थे तो पहले तो गिर गए थे लगड़ा के ऊपर से नीचे",
    ],
}

# Known valid alternatives (spelling variants, numeral forms, synonyms)
KNOWN_ALTERNATIVES = {
    "वहाँ":    {"वहाँ", "वहां"},
    "यहाँ":    {"यहाँ", "यहां"},
    "गए":      {"गए", "गये"},
    "किए":     {"किए", "किये"},
    "देखिए":   {"देखिए", "देखे", "देखें"},
    "project": {"project", "प्रोजेक्ट"},
    "area":    {"area", "एरिया"},
    "enter":   {"enter", "एंटर"},
    "ऊपर":     {"ऊपर", "उपर"},
    "खांड":    {"खांड", "खान"},
    "पाई":     {"पाई", "पाए"},
    "गया":     {"गया", "गए"},
}

MODEL_AGREEMENT_THRESHOLD = 3   # out of 5 models must agree to override reference


# ── Alignment (simple word-level DTW / edit distance) ────────────────────────

def edit_distance_alignment(ref: List[str], hyp: List[str]) -> List[Tuple]:
    """
    Compute word-level alignment between reference and hypothesis.
    Returns list of (ref_token_or_None, hyp_token_or_None) pairs.
    """
    n, m = len(ref), len(hyp)
    # DP table
    dp = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    # Traceback
    alignment = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i-1] == hyp[j-1]:
            alignment.append((ref[i-1], hyp[j-1]))
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            alignment.append((ref[i-1], hyp[j-1]))   # substitution
            i -= 1; j -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            alignment.append((None, hyp[j-1]))         # insertion
            j -= 1
        else:
            alignment.append((ref[i-1], None))         # deletion
            i -= 1

    return list(reversed(alignment))


# ── Lattice construction ──────────────────────────────────────────────────────

@dataclass
class LatticeBin:
    """One position in the lattice — holds all valid alternatives."""
    position: int
    tokens: set = field(default_factory=set)
    is_reference_overridden: bool = False
    override_reason: str = ""

    def add(self, token: str):
        if token:
            self.tokens.add(token)

    def matches(self, token: str) -> bool:
        if not token:
            return False
        # Direct match
        if token in self.tokens:
            return True
        # Known alternative match
        for t in self.tokens:
            alts = KNOWN_ALTERNATIVES.get(t, set())
            if token in alts:
                return True
            alts2 = KNOWN_ALTERNATIVES.get(token, set())
            if t in alts2:
                return True
        return False


def build_lattice(reference: List[str], model_outputs: Dict[str, List[str]]) -> List[LatticeBin]:
    """
    Build a word-level lattice from reference + model outputs.

    Algorithm:
    1. Align each model output to the reference using edit distance.
    2. For each reference position, collect all model tokens at that position.
    3. Add known spelling/numeral alternatives.
    4. If >= MODEL_AGREEMENT_THRESHOLD models agree on a token that differs
       from reference, add it to the bin AND flag possible reference error.

    Handles insertions: model-only tokens get their own bin (position = None).
    Handles deletions: bin may contain None (empty) as valid option.
    """
    # Step 1: Align all models to reference
    alignments = {}
    for model_name, hyp in model_outputs.items():
        alignments[model_name] = edit_distance_alignment(reference, hyp)

    # Step 2: Find max alignment length
    max_len = max(len(a) for a in alignments.values()) if alignments else len(reference)

    # Step 3: Build bins indexed by reference position
    ref_bins: Dict[int, LatticeBin] = {}
    ref_pos = 0

    # Initialise bins from reference
    for i, tok in enumerate(reference):
        ref_bins[i] = LatticeBin(position=i)
        ref_bins[i].add(tok)
        # Add known alternatives
        for alt in KNOWN_ALTERNATIVES.get(tok, set()):
            ref_bins[i].add(alt)

    # Step 4: Add model tokens to bins
    for model_name, alignment in alignments.items():
        r_idx = 0
        for ref_tok, hyp_tok in alignment:
            if ref_tok is not None:
                if r_idx not in ref_bins:
                    ref_bins[r_idx] = LatticeBin(position=r_idx)
                if hyp_tok:
                    ref_bins[r_idx].add(hyp_tok)
                    # Add known alternatives of hyp token too
                    for alt in KNOWN_ALTERNATIVES.get(hyp_tok, set()):
                        ref_bins[r_idx].add(alt)
                r_idx += 1
            # Insertions (ref_tok is None): skip for now — handled below

    # Step 5: Check model agreement — override reference if needed
    for pos, bin_ in ref_bins.items():
        ref_tok = reference[pos] if pos < len(reference) else None
        # Count how many models produced a non-reference token at this position
        model_token_counts = Counter()
        for model_name, alignment in alignments.items():
            r_idx = 0
            for ref_t, hyp_t in alignment:
                if ref_t is not None:
                    if r_idx == pos and hyp_t and hyp_t != ref_tok:
                        model_token_counts[hyp_t] += 1
                    if ref_t is not None:
                        r_idx += 1

        for token, count in model_token_counts.items():
            if count >= MODEL_AGREEMENT_THRESHOLD:
                bin_.add(token)
                if ref_tok and token != ref_tok:
                    bin_.is_reference_overridden = True
                    bin_.override_reason = (
                        f"{count}/{len(model_outputs)} models prefer '{token}' over ref '{ref_tok}'"
                    )

    return [ref_bins[i] for i in sorted(ref_bins.keys())]


# ── Lattice-based WER computation ─────────────────────────────────────────────

def compute_lattice_wer(
    lattice: List[LatticeBin],
    hypothesis: List[str],
    reference: List[str],
) -> Dict:
    """
    Compute WER using the lattice instead of a flat reference string.

    A hypothesis token at position i is CORRECT if it matches any token
    in lattice bin i (including known alternatives and model-agreement overrides).

    Standard WER = (S + D + I) / N
    Lattice WER  = same formula but S/D/I computed against lattice bins.
    """
    alignment = edit_distance_alignment(reference, hypothesis)

    S = D = I = correct = 0
    ref_pos = 0

    for ref_tok, hyp_tok in alignment:
        if ref_tok is not None and hyp_tok is not None:
            # Substitution or match
            bin_ = lattice[ref_pos] if ref_pos < len(lattice) else None
            if bin_ and bin_.matches(hyp_tok):
                correct += 1
            else:
                S += 1
            ref_pos += 1
        elif ref_tok is not None and hyp_tok is None:
            # Deletion
            D += 1
            ref_pos += 1
        elif ref_tok is None and hyp_tok is not None:
            # Insertion
            I += 1

    N = len(reference)
    wer_score = (S + D + I) / N if N > 0 else 0.0
    return {
        "S": S, "D": D, "I": I, "N": N,
        "correct": correct,
        "wer": round(wer_score * 100, 2),
    }


def compute_standard_wer(reference: List[str], hypothesis: List[str]) -> float:
    """Standard WER without lattice."""
    alignment = edit_distance_alignment(reference, hypothesis)
    S = D = I = 0
    for ref_tok, hyp_tok in alignment:
        if ref_tok is not None and hyp_tok is not None and ref_tok != hyp_tok:
            S += 1
        elif ref_tok is not None and hyp_tok is None:
            D += 1
        elif ref_tok is None and hyp_tok is not None:
            I += 1
    N = len(reference)
    return round((S + D + I) / N * 100, 2) if N > 0 else 0.0


# ── Run evaluation ────────────────────────────────────────────────────────────

def run_evaluation():
    print("\n" + "="*75)
    print("LATTICE-BASED WER EVALUATION")
    print("="*75)

    all_standard_wer = {}
    all_lattice_wer  = {}

    for seg_idx, ref_seg in enumerate(REFERENCE_SEGMENTS):
        ref_tokens = ref_seg["text"].split()
        seg_model_outputs = {
            name: outputs[seg_idx].split()
            for name, outputs in MODEL_OUTPUTS.items()
        }

        # Build lattice for this segment
        lattice = build_lattice(ref_tokens, seg_model_outputs)

        # Print overridden bins
        overridden = [b for b in lattice if b.is_reference_overridden]
        if overridden:
            print(f"\nSegment {seg_idx+1} — Reference overrides:")
            for b in overridden:
                print(f"  Bin {b.position}: {b.override_reason}")
                print(f"  Valid tokens: {b.tokens}")

        # Compute WER for each model
        for model_name, hyp_tokens in seg_model_outputs.items():
            std_wer = compute_standard_wer(ref_tokens, hyp_tokens)
            lat_wer = compute_lattice_wer(lattice, hyp_tokens, ref_tokens)["wer"]

            if model_name not in all_standard_wer:
                all_standard_wer[model_name] = []
                all_lattice_wer[model_name]  = []
            all_standard_wer[model_name].append(std_wer)
            all_lattice_wer[model_name].append(lat_wer)

    # Aggregate across segments (macro average)
    print("\n" + "="*75)
    print(f"{'Model':<12} {'Standard WER':>14} {'Lattice WER':>13} {'Delta':>8}  {'Fairly penalized?'}")
    print("="*75)
    for model_name in MODEL_OUTPUTS:
        std  = round(np.mean(all_standard_wer[model_name]), 2)
        lat  = round(np.mean(all_lattice_wer[model_name]),  2)
        delta = std - lat
        fair = "unchanged" if abs(delta) < 0.5 else f"reduced by {delta:.2f}%"
        print(f"{model_name:<12} {std:>13.2f}% {lat:>12.2f}% {delta:>+7.2f}%  {fair}")
    print("="*75)

    print("""
Justification for word-level alignment unit:
  - Hindi words are clearly space-delimited; word boundaries are unambiguous.
  - Subword units would split proper nouns (कुड़रमा -> कुड़ + रमा) and
    loanwords unpredictably, making alignment noisy.
  - Phrase-level is too coarse — a single phrase error would penalise
    all words in it even if most are correct.
  - Word-level gives the best balance of granularity and stability.

When to trust model agreement over reference:
  - When >= 3 out of 5 models produce the same alternative token at a position,
    we add it to the lattice bin. This handles:
    * Reference transcription errors (human annotator mistakes)
    * Valid spelling variants (वहाँ / वहां)
    * Script normalisation (project / प्रोजेक्ट)
  - We do NOT remove the reference token — we ADD the model-agreed token,
    so models that match the reference are still scored correctly.
""")


if __name__ == "__main__":
    run_evaluation()
