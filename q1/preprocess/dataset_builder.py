"""
Q1a — Dataset builder: fetches, processes, and saves HuggingFace dataset.
Builds manifest CSV and HF Dataset from GCP audio + transcription URLs.
"""

import pandas as pd
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from datasets import Dataset, Audio

from q1.preprocess.url_builder import build_url, fetch_json, fetch_audio_bytes
from q1.preprocess.audio_processor import load_and_resample, duration_ok, get_duration
from q1.preprocess.text_normalizer import normalize_text, extract_text_from_transcription

AUDIO_DIR = Path("data/audio")
OUT_DIR   = Path("data/processed")
TARGET_SR = 16_000

AUDIO_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def process_sample(user_id: str, recording_id: str) -> dict | None:
    """Download, validate, and preprocess one sample."""
    trans_url = build_url(user_id, recording_id, "transcription")
    audio_url = build_url(user_id, recording_id, "audio")

    trans_data = fetch_json(trans_url)
    if trans_data is None:
        return None

    text = normalize_text(extract_text_from_transcription(trans_data))
    if not text:
        return None

    audio_bytes = fetch_audio_bytes(audio_url)
    if audio_bytes is None:
        return None

    wav, sr = load_and_resample(audio_bytes)
    if wav is None or not duration_ok(wav, sr):
        return None

    out_path = AUDIO_DIR / f"{user_id}_{recording_id}.wav"
    sf.write(str(out_path), wav, sr)

    return {
        "user_id":      user_id,
        "recording_id": recording_id,
        "audio":        str(out_path),
        "text":         text,
        "duration":     get_duration(wav, sr),
    }


def build_hf_dataset(manifest_df: pd.DataFrame, max_samples: int = None) -> Dataset:
    """Run full preprocessing pipeline and return HuggingFace Dataset."""
    records = []
    rows = list(manifest_df.iterrows())
    if max_samples:
        rows = rows[:max_samples]

    for _, row in tqdm(rows, desc="Preprocessing"):
        result = process_sample(str(row["user_id"]), str(row["recording_id"]))
        if result:
            records.append(result)

    print(f"\nKept {len(records)} / {len(manifest_df)} samples after filtering")

    df = pd.DataFrame(records)
    df.to_csv(OUT_DIR / "processed_manifest.csv", index=False)

    hf_ds = Dataset.from_pandas(df[["audio", "text", "duration"]])
    hf_ds = hf_ds.cast_column("audio", Audio(sampling_rate=TARGET_SR))
    hf_ds.save_to_disk(str(OUT_DIR / "hf_dataset"))

    print(f"Total hours: {df['duration'].sum()/3600:.2f}h")
    print(f"Avg duration: {df['duration'].mean():.2f}s")
    return hf_ds
