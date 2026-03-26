"""
Q1b — Whisper-small fine-tuning trainer.
Configures Seq2SeqTrainer with WER metric for Hindi ASR.
"""

import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate

MODEL_ID  = "openai/whisper-small"
LANGUAGE  = "Hindi"
TASK      = "transcribe"
OUT_DIR   = "models/whisper-small-hindi"

wer_metric = evaluate.load("wer")


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


def get_training_args(fp16: bool = True) -> Seq2SeqTrainingArguments:
    return Seq2SeqTrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=4000,
        gradient_checkpointing=True,
        fp16=fp16,
        evaluation_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=500,
        eval_steps=500,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
    )


def make_compute_metrics(tokenizer):
    def compute_metrics(pred):
        pred_ids  = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_str  = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": 100 * wer_metric.compute(predictions=pred_str, references=label_str)}
    return compute_metrics
