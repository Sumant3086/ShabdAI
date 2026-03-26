"""
Q1c — Model evaluator on FLEURS Hindi test set.
Computes WER for baseline and fine-tuned Whisper-small.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset, Audio
from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizer
import evaluate

MODEL_ID   = "openai/whisper-small"
wer_metric = evaluate.load("wer")


def load_fleurs_test():
    """Load FLEURS Hindi test split."""
    ds = load_dataset("google/fleurs", "hi_in", split="test", trust_remote_code=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
    return ds.rename_column("transcription", "text")


def evaluate_model(model, dataset, feature_extractor, tokenizer, desc="Evaluating"):
    """Run inference and compute WER on a dataset."""
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    all_preds, all_refs = [], []

    for batch in DataLoader(dataset, batch_size=8):
        audio_arrays = [a["array"] for a in batch["audio"]]
        refs         = batch["text"]

        inputs = feature_extractor(
            audio_arrays, sampling_rate=16_000, return_tensors="pt", padding=True
        ).to(device)

        with torch.no_grad():
            predicted_ids = model.generate(
                inputs.input_features, language="hi", task="transcribe"
            )

        preds = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
        all_preds.extend(preds)
        all_refs.extend(refs)

    wer = 100 * wer_metric.compute(predictions=all_preds, references=all_refs)
    print(f"\n{desc} WER: {wer:.2f}%")
    return wer, all_preds, all_refs


def print_wer_table(baseline_wer: float, finetuned_wer: float):
    delta = baseline_wer - finetuned_wer
    print("\n" + "="*55)
    print(f"{'Model':<30} {'WER on FLEURS-hi test':>20}")
    print("="*55)
    print(f"{'Whisper-small (baseline)':<30} {baseline_wer:>19.2f}%")
    print(f"{'Whisper-small (fine-tuned)':<30} {finetuned_wer:>19.2f}%")
    print("-"*55)
    print(f"{'Improvement':<30} {delta:>+19.2f}%")
    print("="*55)
