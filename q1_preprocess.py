"""
Q1a — Dataset Preprocessing Entry Point

Reads data/training_manifest_valid.csv (90 records, 17.78h of Hindi audio),
fetches audio + transcriptions from GCP, normalises text, resamples audio,
and exports a HuggingFace Dataset ready for Whisper fine-tuning.

Run:
    python q1_preprocess.py
"""

import pandas as pd
from pathlib import Path
from q1.preprocess.dataset_builder import build_hf_dataset

MANIFEST = "data/training_manifest_valid.csv"


def main():
    if not Path(MANIFEST).exists():
        print(f"[ERROR] {MANIFEST} not found.")
        print("Run decode_urls.py and verify_manifest.py first, or")
        print("ensure TrainingData.pdf is in the project root.")
        return

    df = pd.read_csv(MANIFEST)
    print(f"Loaded manifest: {len(df)} records, "
          f"{df['duration_sec'].sum()/3600:.2f} hours total")

    # Rename columns to match dataset_builder expectations
    df = df.rename(columns={"folder_id": "user_id"})

    hf_ds = build_hf_dataset(df)
    print(f"\nDone. HuggingFace Dataset saved to data/processed/hf_dataset/")
    print(f"Run q1_finetune.py next to fine-tune Whisper-small.")


if __name__ == "__main__":
    main()
