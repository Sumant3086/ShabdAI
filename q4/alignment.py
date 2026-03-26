"""
Q4 — Word-level edit distance alignment.
Aligns reference and hypothesis token sequences using dynamic programming.
Returns list of (ref_token, hyp_token) pairs covering matches,
substitutions, insertions, and deletions.

Alignment unit: WORD
Justification:
  - Hindi words are clearly space-delimited; boundaries are unambiguous.
  - Subword units would fragment proper nouns (कुड़रमा -> कुड़ + रमा)
    and loanwords unpredictably, making alignment noisy.
  - Phrase-level is too coarse — one phrase error penalises all words in it.
  - Word-level gives the best balance of granularity and stability.
"""

import numpy as np
from typing import List, Tuple, Optional


def edit_distance_alignment(
    ref: List[str], hyp: List[str]
) -> List[Tuple[Optional[str], Optional[str]]]:
    """
    Compute word-level alignment between reference and hypothesis.

    Returns list of (ref_token_or_None, hyp_token_or_None):
      (word, word)  -> match or substitution
      (word, None)  -> deletion
      (None, word)  -> insertion
    """
    n, m = len(ref), len(hyp)
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
            alignment.append((ref[i-1], hyp[j-1])); i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            alignment.append((ref[i-1], hyp[j-1])); i -= 1; j -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            alignment.append((None, hyp[j-1])); j -= 1
        else:
            alignment.append((ref[i-1], None)); i -= 1

    return list(reversed(alignment))


def compute_standard_wer(reference: List[str], hypothesis: List[str]) -> float:
    """Standard WER without lattice."""
    alignment = edit_distance_alignment(reference, hypothesis)
    S = D = I = 0
    for ref_tok, hyp_tok in alignment:
        if ref_tok and hyp_tok and ref_tok != hyp_tok:
            S += 1
        elif ref_tok and not hyp_tok:
            D += 1
        elif not ref_tok and hyp_tok:
            I += 1
    N = len(reference)
    return round((S + D + I) / N * 100, 2) if N > 0 else 0.0
