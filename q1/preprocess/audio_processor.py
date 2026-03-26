"""
Q1a — Audio processing utilities.
Loads audio bytes, converts to mono, resamples to 16kHz.
Filters by duration for Whisper compatibility.
"""

import io
import numpy as np
import librosa
import soundfile as sf

TARGET_SR    = 16_000
MAX_DURATION = 30.0   # Whisper hard limit (seconds)
MIN_DURATION =  0.5   # skip very short clips


def load_and_resample(audio_bytes: bytes, target_sr: int = TARGET_SR):
    """
    Load audio from raw bytes, convert to mono, resample to target_sr.
    Returns (waveform_np_float32, sample_rate) or (None, None) on failure.
    """
    try:
        wav, sr = sf.read(io.BytesIO(audio_bytes))
    except Exception:
        try:
            wav, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
        except Exception as e:
            print(f"  [WARN] Audio decode failed: {e}")
            return None, None

    # Convert to mono
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    # Resample if needed
    if sr != target_sr:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=target_sr)

    return wav.astype(np.float32), target_sr


def duration_ok(wav: np.ndarray, sr: int) -> bool:
    """Check if audio duration is within acceptable bounds."""
    dur = len(wav) / sr
    return MIN_DURATION <= dur <= MAX_DURATION


def get_duration(wav: np.ndarray, sr: int) -> float:
    return round(len(wav) / sr, 2)
