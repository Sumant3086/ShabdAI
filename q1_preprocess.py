"""
Q1a — Dataset Preprocessing for Hindi ASR Training
Fetches audio + transcription from GCP URLs, applies cleaning,
resamples to 16kHz, and saves a HuggingFace-compatible dataset.
"""

import os
import re
import json
import requests
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from pathlib import Path
from tqdm import tqdm
from datasets import Dataset, Audio

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL   = "https://storage.googleapis.com/upload_goai"
DATA_CSV   = "dataset_index.csv"   # local manifest (see build_manifest below)
OUT_DIR    = Path("data/processed")
AUDIO_DIR  = Path("data/audio")
TARGET_SR  = 16_000
MAX_DURATION = 30.0   # seconds — Whisper's hard limit
MIN_DURATION =  0.5   # skip very short clips

OUT_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def build_url(user_id: str, recording_id: str, kind: str) -> str:
    """
    Reconstruct GCP URL from the pattern:
      https://storage.googleapis.com/upload_goai/{user_id}/{recording_id}_{kind}.json
    kind ∈ {'transcription', 'metadata'}  |  audio has no suffix
    """
    if kind == "audio":
        return f"{BASE_URL}/{user_id}/{recording_id}.wav"
    return f"{BASE_URL}/{user_id}/{recording_id}_{kind}.json"


def fetch_json(url: str) -> dict | None:
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  [WARN] Could not fetch {url}: {e}")
        return None


def fetch_audio_bytes(url: str) -> bytes | None:
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.content
    except Exception as e:
        print(f"  [WARN] Could not fetch audio {url}: {e}")
        return None


# ── Text Normalisation ────────────────────────────────────────────────────────

# Characters to strip (punctuation that Whisper doesn't need)
_STRIP_CHARS = re.compile(r'[।|॥,\.\!\?\"\'\(\)\[\]\{\};:—–\-]')
# Collapse multiple spaces
_MULTI_SPACE = re.compile(r'\s+')

def normalize_text(text: str) -> str:
    """
    Light normalisation for Hindi transcripts:
    1. Strip danda / double-danda and common punctuation
    2. Normalise Unicode (NFC)
    3. Collapse whitespace
    4. Strip leading/trailing space
    We intentionally keep Devanagari digits and English words in Devanagari
    (per transcription guidelines).
    """
    import unicodedata
    text = unicodedata.normalize("NFC", text)
    text = _STRIP_CHARS.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


# ── Audio Processing ──────────────────────────────────────────────────────────

def load_and_resample(audio_bytes: bytes, target_sr: int = TARGET_SR):
    """
    Load audio from raw bytes, convert to mono, resample to target_sr.
    Returns (waveform_np_float32, sample_rate) or (None, None) on failure.
    """
    import io
    try:
        wav, sr = sf.read(io.BytesIO(audio_bytes))
    except Exception:
        try:
            import io as _io
            wav, sr = librosa.load(_io.BytesIO(audio_bytes), sr=None, mono=True)
        except Exception as e:
            print(f"  [WARN] Audio decode failed: {e}")
            return None, None

    # Mono
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    # Resample
    if sr != target_sr:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=target_sr)

    return wav.astype(np.float32), target_sr


def duration_ok(wav: np.ndarray, sr: int) -> bool:
    dur = len(wav) / sr
    return MIN_DURATION <= dur <= MAX_DURATION


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def process_sample(user_id: str, recording_id: str) -> dict | None:
    """
    Download, validate, and preprocess one sample.
    Returns a dict ready for HuggingFace Dataset, or None if skipped.
    """
    trans_url  = build_url(user_id, recording_id, "transcription")
    audio_url  = build_url(user_id, recording_id, "audio")

    # 1. Fetch transcription
    trans_data = fetch_json(trans_url)
    if trans_data is None:
        return None

    # Transcription JSON may be a list of segments or a dict with 'text'
    if isinstance(trans_data, list):
        text = " ".join(seg.get("text", "") for seg in trans_data)
    elif isinstance(trans_data, dict):
        text = trans_data.get("text", trans_data.get("transcription", ""))
    else:
        text = str(trans_data)

    text = normalize_text(text)
    if not text:
        return None

    # 2. Fetch audio
    audio_bytes = fetch_audio_bytes(audio_url)
    if audio_bytes is None:
        return None

    wav, sr = load_and_resample(audio_bytes)
    if wav is None or not duration_ok(wav, sr):
        return None

    # 3. Save audio locally (for HF Dataset path reference)
    out_path = AUDIO_DIR / f"{user_id}_{recording_id}.wav"
    sf.write(str(out_path), wav, sr)

    return {
        "user_id":       user_id,
        "recording_id":  recording_id,
        "audio":         str(out_path),
        "text":          text,
        "duration":      round(len(wav) / sr, 2),
    }


def build_manifest_from_example():
    """
    Demonstrates how to build a manifest from the known example URL pattern.
    In practice, you'd parse the original dataset CSV/JSON index.
    """
    # Example entry derived from the provided sample URL:
    # https://storage.googleapis.com/upload_goai/967179/825780_transcription.json
    examples = [
        {"user_id": "967179", "recording_id": "825780"},
        # Add more entries here from the dataset index
    ]
    return pd.DataFrame(examples)


def run_preprocessing(manifest_df: pd.DataFrame, max_samples: int = None):
    records = []
    iterable = manifest_df.iterrows()
    if max_samples:
        iterable = list(iterable)[:max_samples]

    for _, row in tqdm(iterable, desc="Preprocessing"):
        result = process_sample(str(row["user_id"]), str(row["recording_id"]))
        if result:
            records.append(result)

    print(f"\n✓ Kept {len(records)} / {len(manifest_df)} samples after filtering")

    df = pd.DataFrame(records)
    df.to_csv(OUT_DIR / "processed_manifest.csv", index=False)

    # Build HuggingFace Dataset
    hf_dataset = Dataset.from_pandas(df[["audio", "text", "duration"]])
    hf_dataset = hf_dataset.cast_column("audio", Audio(sampling_rate=TARGET_SR))
    hf_dataset.save_to_disk(str(OUT_DIR / "hf_dataset"))

    print(f"✓ HF Dataset saved to {OUT_DIR / 'hf_dataset'}")
    print(f"\nPreprocessing summary:")
    print(f"  Total samples:    {len(records)}")
    print(f"  Avg duration:     {df['duration'].mean():.2f}s")
    print(f"  Total hours:      {df['duration'].sum()/3600:.2f}h")
    print(f"  Min/Max duration: {df['duration'].min():.2f}s / {df['duration'].max():.2f}s")

    return hf_dataset


if __name__ == "__main__":
    manifest = build_manifest_from_example()
    run_preprocessing(manifest)
