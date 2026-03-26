"""
Decode the interleaved GCP URLs from the PDF.

Observed pattern for known entry (folder=967179, recording=825780):
  garbled segment: 'oe9sc6ht7iot1an7lk/9hs/q-8d_2ad5taa7t-8ac0'

The folder ID 967179 and recording 825780 are interleaved with filler chars.
Let's extract ONLY the digit characters in order:
  '9sc6ht7iot1an7lk/9hs/q-8d_2ad5taa7t-8ac0'
  digits only: 9 6 7 1 7 9 9 8 2 5 7 8 0
  first 6: 967179 = folder_id ✓
  next 6:  825780 = recording_id ✓  (with one extra 9 between)

So the pattern is: all digits in order, first 6 = folder_id, skip 1, next 6 = recording_id
"""
import re, requests, pandas as pd
from pathlib import Path

BASE = "https://storage.googleapis.com/upload_goai"

def decode_garbled_url(garbled, recording_id):
    """
    Extract folder_id from garbled URL.
    All digits in sequence: first 6 = folder_id.
    """
    all_digits = re.sub(r'[^0-9]', '', garbled)
    # Try lengths 4-7 for folder_id
    for length in [6, 5, 4, 7]:
        candidate = all_digits[:length]
        if candidate and candidate != recording_id:
            # Verify: remaining digits should start with recording_id
            remaining = all_digits[length:]
            if recording_id in remaining[:len(recording_id)+2]:
                return candidate
    # Fallback: just take first 6 digits
    return all_digits[:6] if len(all_digits) >= 6 else None

# Test with known example
test = "oe9sc6ht7iot1an7lk/9hs/q-8d_2ad5taa7t-8ac0"
result = decode_garbled_url(test, "825780")
print(f"Test: {result} (expected 967179)")

# Also try: just extract all digits and take first N that aren't the recording_id
def extract_folder_id(garbled, recording_id):
    digits = re.sub(r'[^0-9]', '', garbled)
    # The folder_id appears before the recording_id in the digit stream
    idx = digits.find(recording_id)
    if idx > 0:
        folder = digits[:idx]
        # Take last 4-7 chars of folder (in case there are leading digits)
        for l in [6, 5, 4, 7]:
            if len(folder) >= l:
                return folder[-l:]
    return digits[:6] if len(digits) >= 6 else None

# Test
result2 = extract_folder_id(test, "825780")
print(f"Test2: {result2} (expected 967179)")

# Read parsed rows
with open("training_data_extracted.txt", encoding="utf-8") as f:
    text = f.read()

rows = []
for line in text.split("\n"):
    parts = line.strip().split()
    if len(parts) >= 4:
        if re.match(r'^\d{3,7}$', parts[0]) and re.match(r'^\d{5,7}$', parts[1]):
            try:
                rows.append({
                    "speaker_id":   parts[0],
                    "recording_id": parts[1],
                    "language":     parts[2] if parts[2] in ('hi','en') else 'hi',
                    "duration":     float(parts[3]),
                    "garbled":      " ".join(parts[4:]) if len(parts) > 4 else ""
                })
            except Exception:
                pass

print(f"\nParsed {len(rows)} rows")

# Decode folder IDs
results = []
for row in rows:
    fid = extract_folder_id(row["garbled"], row["recording_id"])
    if fid:
        results.append({
            "folder_id":         fid,
            "speaker_id":        row["speaker_id"],
            "recording_id":      row["recording_id"],
            "language":          row["language"],
            "duration_sec":      row["duration"],
            "transcription_url": f"{BASE}/{fid}/{row['recording_id']}_transcription.json",
            "audio_url":         f"{BASE}/{fid}/{row['recording_id']}.wav",
        })

df = pd.DataFrame(results)
print(f"Decoded {len(df)} records")
if len(df):
    print(df[["folder_id","speaker_id","recording_id","duration_sec"]].head(10).to_string())

# Verify first 10
print("\nVerifying first 10 URLs...")
ok = 0
for _, row in df.head(10).iterrows():
    url = row["transcription_url"]
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            n = len(data) if isinstance(data, list) else 1
            print(f"  OK  folder={row['folder_id']} rec={row['recording_id']} ({n} segs)")
            ok += 1
        else:
            print(f"  {r.status_code} folder={row['folder_id']} rec={row['recording_id']}")
    except Exception as e:
        print(f"  ERR: {e}")

print(f"\n{ok}/10 verified")
Path("data").mkdir(exist_ok=True)
df.to_csv("data/training_manifest.csv", index=False)
print("Saved to data/training_manifest.csv")
