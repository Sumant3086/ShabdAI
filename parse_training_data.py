"""
Parse TrainingData.pdf to extract user_id, recording_id, language, duration
and reconstruct valid GCP URLs using the known pattern.
"""
import pdfplumber, re, json, requests, pandas as pd
from pathlib import Path

PDF = "TrainingData.pdf"
BASE = "https://storage.googleapis.com/upload_goai"

records = []

with pdfplumber.open(PDF) as pdf:
    full_text = ""
    for page in pdf.pages:
        # Use extract_words for better column separation
        words = page.extract_words(x_tolerance=3, y_tolerance=3)
        lines = {}
        for w in words:
            y = round(w['top'], 0)
            lines.setdefault(y, []).append(w['text'])
        for y in sorted(lines):
            full_text += " ".join(lines[y]) + "\n"

# Parse rows: user_id recording_id language duration ...
# Header line: user_id recording_id language duration rec_url_gcp ...
rows = []
for line in full_text.split("\n"):
    parts = line.strip().split()
    if len(parts) >= 4:
        # Check if first two tokens look like numeric IDs
        if re.match(r'^\d{3,7}$', parts[0]) and re.match(r'^\d{5,7}$', parts[1]):
            try:
                user_id      = parts[0]
                recording_id = parts[1]
                language     = parts[2] if parts[2] in ('hi', 'en') else 'hi'
                duration     = float(parts[3]) if re.match(r'^\d+\.?\d*$', parts[3]) else None
                rows.append({
                    "user_id":      user_id,
                    "recording_id": recording_id,
                    "language":     language,
                    "duration":     duration,
                })
            except Exception:
                pass

print(f"Parsed {len(rows)} records from PDF")

# Reconstruct valid URLs
for r in rows:
    uid = r["user_id"]
    rid = r["recording_id"]
    r["transcription_url"] = f"{BASE}/{uid}/{rid}_transcription.json"
    r["audio_url"]         = f"{BASE}/{uid}/{rid}.wav"
    r["metadata_url"]      = f"{BASE}/{uid}/{rid}_metadata.json"

df = pd.DataFrame(rows)
print(df.head(10).to_string())
print(f"\nTotal records: {len(df)}")
print(f"Total duration: {df['duration'].sum()/3600:.2f} hours")

# Save manifest
Path("data").mkdir(exist_ok=True)
df.to_csv("data/training_manifest.csv", index=False)
print(f"\nSaved to data/training_manifest.csv")

# Verify first 3 URLs are reachable
print("\nVerifying first 3 transcription URLs...")
for _, row in df.head(3).iterrows():
    url = row["transcription_url"]
    try:
        r = requests.get(url, timeout=10)
        status = f"HTTP {r.status_code}"
        if r.status_code == 200:
            data = r.json()
            n_segs = len(data) if isinstance(data, list) else 1
            status += f" — {n_segs} segments"
    except Exception as e:
        status = f"ERROR: {e}"
    print(f"  {url}")
    print(f"  -> {status}\n")
