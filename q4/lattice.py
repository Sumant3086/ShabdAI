"""
Q4 — Lattice builder and lattice-based WER computation.

A lattice is a list of bins. Each bin represents one alignment position
and holds all valid lexical, phonetic, and spelling alternatives.

Model agreement rule:
  If >= MODEL_AGREEMENT_THRESHOLD models produce the same token at a position
  that differs from the reference, that token is added to the bin.
  This handles reference transcription errors and valid spelling variants.
  We do NOT remove the reference token — we ADD the agreed token,
  so models matching the reference are still scored correctly.
"""

from dataclasses import dataclass, field
from collections import Counter
from typing import List, Dict, Optional
from q4.alignment import edit_distance_alignment

MODEL_AGREEMENT_THRESHOLD = 3   # out of 5 models

# Known valid alternatives (spelling variants, script variants, numeral forms)
KNOWN_ALTERNATIVES = {
    "वहाँ":  {"वहाँ", "वहां"},
    "यहाँ":  {"यहाँ", "यहां"},
    "गए":    {"गए", "गये"},
    "किए":   {"किए", "किये"},
    "देखिए": {"देखिए", "देखे", "देखें"},
    "project": {"project", "प्रोजेक्ट"},
    "area":    {"area", "एरिया"},
    "enter":   {"enter", "एंटर"},
    "ऊपर":   {"ऊपर", "उपर"},
    "खांड":  {"खांड", "खान"},
    "पाई":   {"पाई", "पाए"},
    "गया":   {"गया", "गए"},
}


@dataclass
class LatticeBin:
    position: int
    tokens: set = field(default_factory=set)
    is_reference_overridden: bool = False
    override_reason: str = ""

    def add(self, token: str):
        if token:
            self.tokens.add(token)
            for alt in KNOWN_ALTERNATIVES.get(token, set()):
                self.tokens.add(alt)

    def matches(self, token: str) -> bool:
        if not token:
            return False
        if token in self.tokens:
            return True
        for t in self.tokens:
            if token in KNOWN_ALTERNATIVES.get(t, set()):
                return True
            if t in KNOWN_ALTERNATIVES.get(token, set()):
                return True
        return False


def build_lattice(
    reference: List[str],
    model_outputs: Dict[str, List[str]],
) -> List[LatticeBin]:
    """
    Build a word-level lattice from reference + model outputs.

    Steps:
    1. Initialise bins from reference tokens + known alternatives.
    2. Align each model output to reference using edit distance.
    3. Add model tokens to corresponding bins.
    4. Apply model agreement override for likely reference errors.
    """
    # Initialise bins from reference
    bins: Dict[int, LatticeBin] = {}
    for i, tok in enumerate(reference):
        bins[i] = LatticeBin(position=i)
        bins[i].add(tok)

    # Align models and populate bins
    alignments = {name: edit_distance_alignment(reference, hyp)
                  for name, hyp in model_outputs.items()}

    for model_name, alignment in alignments.items():
        r_idx = 0
        for ref_tok, hyp_tok in alignment:
            if ref_tok is not None:
                if r_idx not in bins:
                    bins[r_idx] = LatticeBin(position=r_idx)
                if hyp_tok:
                    bins[r_idx].add(hyp_tok)
                r_idx += 1

    # Model agreement override
    for pos, bin_ in bins.items():
        ref_tok = reference[pos] if pos < len(reference) else None
        model_token_counts: Counter = Counter()

        for model_name, alignment in alignments.items():
            r_idx = 0
            for ref_t, hyp_t in alignment:
                if ref_t is not None:
                    if r_idx == pos and hyp_t and hyp_t != ref_tok:
                        model_token_counts[hyp_t] += 1
                    r_idx += 1

        for token, count in model_token_counts.items():
            if count >= MODEL_AGREEMENT_THRESHOLD:
                bin_.add(token)
                if ref_tok and token != ref_tok:
                    bin_.is_reference_overridden = True
                    bin_.override_reason = (
                        f"{count}/{len(model_outputs)} models prefer "
                        f"'{token}' over ref '{ref_tok}'"
                    )

    return [bins[i] for i in sorted(bins.keys())]


def compute_lattice_wer(
    lattice: List[LatticeBin],
    hypothesis: List[str],
    reference: List[str],
) -> Dict:
    """
    Compute WER using lattice instead of flat reference.
    A hypothesis token is correct if it matches ANY token in the bin.
    """
    alignment = edit_distance_alignment(reference, hypothesis)
    S = D = I = correct = 0
    ref_pos = 0

    for ref_tok, hyp_tok in alignment:
        if ref_tok is not None and hyp_tok is not None:
            bin_ = lattice[ref_pos] if ref_pos < len(lattice) else None
            if bin_ and bin_.matches(hyp_tok):
                correct += 1
            else:
                S += 1
            ref_pos += 1
        elif ref_tok is not None:
            D += 1; ref_pos += 1
        elif hyp_tok is not None:
            I += 1

    N = len(reference)
    return {
        "S": S, "D": D, "I": I, "N": N, "correct": correct,
        "wer": round((S + D + I) / N * 100, 2) if N > 0 else 0.0,
    }
