"""
Q1b/c — Fine-tune Whisper-small on Hindi dataset + evaluate on FLEURS
Reports WER for baseline and fine-tuned model.
"""

import os
import torch
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from datasets import load_dataset, load_from_disk, Audio, DatasetDict
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID      = "openai/whisper-small"
LANGUAGE      = "Hindi"
TASK          = "transcribe"
OUT_DIR       = "models/whisper-small-hindi"
PROCESSED_DIR = "data/processed/hf_dataset"
BATCH_SIZE    = 16
GRAD_ACCUM    = 2
LR            = 1e-5
WARMUP_STEPS  = 500
MAX_STEPS     = 4000
EVAL_STEPS    = 500
SAVE_STEPS    = 500
FP16          = torch.cuda.is_available()

# ── Load processor ────────────────────────────────────────────────────────────
feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_ID)
tokenizer = WhisperTokenizer.from_pretrained(MODEL_ID, language=LANGUAGE, task=TASK)
processor = WhisperProcessor.from_pretrained(MODEL_ID, language=LANGUAGE, task=TASK)

# ── Data Collator ─────────────────────────────────────────────────────────────
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch   = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ── Preprocessing ─────────────────────────────────────────────────────────────
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = tokenizer(batch["text"]).input_ids
    return batch


# ── Metrics ───────────────────────────────────────────────────────────────────
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids   = pred.predictions
    label_ids  = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str  = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


# ── Load & split dataset ──────────────────────────────────────────────────────
def load_training_data():
    ds = load_from_disk(PROCESSED_DIR)
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
    split = ds.train_test_split(test_size=0.1, seed=42)
    return DatasetDict({"train": split["train"], "validation": split["test"]})


# ── Load FLEURS Hindi test set ────────────────────────────────────────────────
def load_fleurs_test():
    fleurs = load_dataset("google/fleurs", "hi_in", split="test", trust_remote_code=True)
    fleurs = fleurs.cast_column("audio", Audio(sampling_rate=16_000))
    # FLEURS uses 'transcription' column
    fleurs = fleurs.rename_column("transcription", "text")
    return fleurs


# ── Evaluate a model on a dataset ────────────────────────────────────────────
def evaluate_model(model, dataset, desc="Evaluating"):
    from torch.utils.data import DataLoader
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    all_preds, all_refs = [], []

    for batch in DataLoader(dataset, batch_size=8):
        audio_arrays = [a["array"] for a in batch["audio"]]
        refs         = batch["text"]

        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=16_000,
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            predicted_ids = model.generate(
                inputs.input_features,
                language="hi",
                task="transcribe",
            )

        preds = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
        all_preds.extend(preds)
        all_refs.extend(refs)

    wer = 100 * wer_metric.compute(predictions=all_preds, references=all_refs)
    print(f"\n{desc} WER: {wer:.2f}%")
    return wer, all_preds, all_refs


# ── Fine-tuning ───────────────────────────────────────────────────────────────
def finetune():
    dataset = load_training_data()
    dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=4)

    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    model.generation_config.language = LANGUAGE
    model.generation_config.task     = TASK
    model.generation_config.forced_decoder_ids = None

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS,
        gradient_checkpointing=True,
        fp16=FP16,
        evaluation_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
    trainer.save_model(OUT_DIR)
    processor.save_pretrained(OUT_DIR)
    print(f"\n✓ Fine-tuned model saved to {OUT_DIR}")
    return model


# ── WER Report Table ──────────────────────────────────────────────────────────
def print_wer_table(baseline_wer, finetuned_wer):
    print("\n" + "="*55)
    print(f"{'Model':<30} {'WER on FLEURS-hi test':>20}")
    print("="*55)
    print(f"{'Whisper-small (baseline)':<30} {baseline_wer:>19.2f}%")
    print(f"{'Whisper-small (fine-tuned)':<30} {finetuned_wer:>19.2f}%")
    delta = baseline_wer - finetuned_wer
    print("-"*55)
    print(f"{'Improvement':<30} {delta:>+19.2f}%")
    print("="*55)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fleurs_test = load_fleurs_test()

    # Baseline evaluation
    print("\n── Baseline Whisper-small ──")
    baseline_model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    baseline_wer, _, _ = evaluate_model(baseline_model, fleurs_test, "Baseline")

    # Fine-tune
    print("\n── Fine-tuning ──")
    finetuned_model = finetune()

    # Fine-tuned evaluation
    print("\n── Fine-tuned Whisper-small ──")
    finetuned_wer, ft_preds, ft_refs = evaluate_model(
        finetuned_model, fleurs_test, "Fine-tuned"
    )

    print_wer_table(baseline_wer, finetuned_wer)

    # Save predictions for error analysis (Q1d/e)
    import pandas as pd
    pd.DataFrame({"reference": ft_refs, "hypothesis": ft_preds}).to_csv(
        "data/finetuned_fleurs_predictions.csv", index=False
    )
    print("\n✓ Predictions saved to data/finetuned_fleurs_predictions.csv")
