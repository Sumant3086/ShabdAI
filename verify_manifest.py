"""Verify all 104 URLs and build final clean manifest."""
import requests, pandas as pd
from tqdm import tqdm

df = pd.read_csv("data/training_manifest.csv")
print(f"Total records: {len(df)}")

valid, invalid = [], []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Verifying"):
    url = row["transcription_url"]
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            n_segs = len(data) if isinstance(data, list) else 1
            total_text = " ".join(s.get("text","") for s in data) if isinstance(data, list) else str(data)
            valid.append({**row.to_dict(), "n_segments": n_segs, "sample_text": total_text[:100]})
        else:
            invalid.append({**row.to_dict(), "error": f"HTTP {r.status_code}"})
    except Exception as e:
        invalid.append({**row.to_dict(), "error": str(e)})

df_valid = pd.DataFrame(valid)
df_invalid = pd.DataFrame(invalid)

print(f"\nValid:   {len(df_valid)}")
print(f"Invalid: {len(df_invalid)}")
print(f"Total duration (valid): {df_valid['duration_sec'].sum()/3600:.2f} hours")

df_valid.to_csv("data/training_manifest_valid.csv", index=False)
if len(df_invalid):
    df_invalid.to_csv("data/training_manifest_invalid.csv", index=False)
    print("\nInvalid entries:")
    print(df_invalid[["folder_id","recording_id","error"]].to_string())

print("\nSample valid entries:")
print(df_valid[["folder_id","recording_id","duration_sec","n_segments"]].head(10).to_string())
