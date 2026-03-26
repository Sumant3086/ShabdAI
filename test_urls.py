"""Test URL patterns against the known working example from the assignment."""
import requests, json

# Known working URL from assignment
KNOWN = "https://storage.googleapis.com/upload_goai/967179/825780_transcription.json"

# Our parsed first record: user_id=245746, recording_id=825780
# Note: recording_id 825780 appears in BOTH the known URL (user 967179)
# and our parsed data (user 245746) — the user_id differs

tests = [
    # Known working
    "https://storage.googleapis.com/upload_goai/967179/825780_transcription.json",
    # Our parsed record 0
    "https://storage.googleapis.com/upload_goai/245746/825780_transcription.json",
    # Try with parsed user_id for known recording
    "https://storage.googleapis.com/upload_goai/245746/825780_transcription.json",
    # Try record 1
    "https://storage.googleapis.com/upload_goai/291038/825727_transcription.json",
    # Try record 2
    "https://storage.googleapis.com/upload_goai/246004/988596_transcription.json",
]

for url in tests:
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            n = len(data) if isinstance(data, list) else 1
            print(f"OK  {url}")
            print(f"    {n} segments, first: {str(data[0])[:80] if isinstance(data, list) else str(data)[:80]}")
        else:
            print(f"{r.status_code} {url}")
    except Exception as e:
        print(f"ERR {url}: {e}")
