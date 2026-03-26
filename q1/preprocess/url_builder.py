"""
Q1a — GCP URL builder and HTTP fetchers.
Reconstructs dataset URLs from user_id + recording_id pattern.
"""

import requests

BASE_URL = "https://storage.googleapis.com/upload_goai"


def build_url(user_id: str, recording_id: str, kind: str) -> str:
    """
    Reconstruct GCP URL.
    kind in {'transcription', 'metadata', 'audio'}
    Example:
      build_url('967179', '825780', 'transcription')
      -> https://storage.googleapis.com/upload_goai/967179/825780_transcription.json
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
