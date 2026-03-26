"""
Q1d — Stratified error sampler.
Samples utterances where fine-tuned model still produces errors,
stratified by CER severity to avoid cherry-picking.
"""

import pandas as pd
from jiwer import wer, cer


def load_predictions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).dropna(subset=["reference", "hypothesis"])
    df["wer_score"] = df.apply(lambda r: wer(r["reference"], r["hypothesis"]), axis=1)
    df["cer_score"] = df.apply(lambda r: cer(r["reference"], r["hypothesis"]), axis=1)
    return df


def sample_errors(df: pd.DataFrame, n_total: int = 30) -> pd.DataFrame:
    """
    Sampling strategy:
    1. Keep only utterances with WER > 0 (actual errors).
    2. Bin by CER: low (0-0.2), medium (0.2-0.5), high (>0.5).
    3. Sample every Nth from each bin proportionally to bin size.

    This ensures coverage across severity levels without cherry-picking.
    Reproducible — no random sampling, purely systematic (every Nth).
    """
    errors = df[df["wer_score"] > 0].copy()
    errors["severity"] = pd.cut(
        errors["cer_score"],
        bins=[0, 0.2, 0.5, float("inf")],
        labels=["low", "medium", "high"],
    )

    sampled = []
    for severity, group in errors.groupby("severity", observed=True):
        n_from_bin = max(1, round(n_total * len(group) / len(errors)))
        step = max(1, len(group) // n_from_bin)
        sampled.append(group.iloc[::step].head(n_from_bin))

    result = pd.concat(sampled).head(n_total).reset_index(drop=True)
    print(f"Sampled {len(result)} error utterances")
    print(result["severity"].value_counts().to_string())
    return result
