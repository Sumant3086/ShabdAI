"""
Q2 -- ASR Cleanup Pipeline: Number Normalisation + English Word Detection

Entry point that wires together q2/number_normalizer.py and q2/english_detector.py,
runs demos, and optionally generates raw ASR transcripts using Whisper-small.
Heavy deps (torch, librosa, transformers) are imported lazily inside
generate_raw_asr() so demos work without a GPU environment.
"""

import pandas as pd
from pathlib import Path

from q2.number_normalizer import normalise_numbers
from q2.english_detector import tag_english_words, get_english_words

MODEL_ID = "openai/whisper-small"
OUT_DIR  = Path("data/q2_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUMBER_EXAMPLES = [
    ("उसने चौदह किताबें खरीदीं",        "उसने 14 किताबें खरीदीं",      "simple"),
    ("तीन सौ चौवन रुपये दिए",           "354 रुपये दिए",                "compound"),
    ("पच्चीस लोग आए थे",                "25 लोग आए थे",                 "two-digit"),
    ("एक हज़ार पाँच सौ मीटर दूर है",    "1500 मीटर दूर है",             "large compound"),
    ("छः सात आठ किलोमीटर में नौ बजे",  "6 7 8 किलोमीटर में 9 बजे",    "sequence"),
    ("दो-चार बातें करनी हैं",           "दो-चार बातें करनी हैं",        "IDIOM: kept as-is"),
    ("एक न एक दिन आएगा",               "एक न एक दिन आएगा",             "IDIOM: kept as-is"),
    ("चार चाँद लगा दिए",                "4 चाँद लगा दिए",               "चार->4 (space, not idiom)"),
]

ENGLISH_DETECTION_EXAMPLES = [
    "मेरा इंटरव्यू बहुत अच्छा गया और मुझे जॉब मिल गई",
    "हमारा प्रोजेक्ट भी था कि जो जनजाति पाई जाती है",
    "हम टेंट गड़ा और रहे",
    "मेरा interview अच्छा गया",
    "ये problem solve नहीं हो रहा",
    "अमेज़न का जंगल होता है",
    "रोड पे होता है न रोड का जो एरिया",
]


def demo_number_normalisation():
    print("\n-- Number Normalisation Examples --")
    print(f"{'INPUT':<45} {'OUTPUT':<45} NOTE")
    print("-" * 110)
    for inp, expected, note in NUMBER_EXAMPLES:
        result = normalise_numbers(inp)
        status = "OK" if result == expected else f"DIFF (got: {result})"
        print(f"{inp:<45} {result:<45} [{note}] {status}")


def demo_english_detection():
    print("\n-- English Word Detection Examples --")
    for text in ENGLISH_DETECTION_EXAMPLES:
        tagged = tag_english_words(text)
        print(f"  IN:  {text}")
        print(f"  OUT: {tagged}")
        print()


def generate_raw_asr(audio_paths, reference_texts):
    import torch
    import librosa
    from tqdm import tqdm
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model     = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    records = []
    for audio_path, ref in tqdm(zip(audio_paths, reference_texts), desc="ASR"):
        try:
            wav, _ = librosa.load(audio_path, sr=16000, mono=True)
            inputs = processor(wav, sampling_rate=16000, return_tensors="pt").to(device)
            with torch.no_grad():
                ids = model.generate(inputs.input_features, language="hi", task="transcribe")
            raw_asr = processor.batch_decode(ids, skip_special_tokens=True)[0]
            records.append({
                "audio_path": audio_path, "reference": ref, "raw_asr": raw_asr,
                "num_norm": normalise_numbers(raw_asr),
                "en_tagged": tag_english_words(raw_asr),
                "both_ops": tag_english_words(normalise_numbers(raw_asr)),
            })
        except Exception as e:
            print(f"  [WARN] {audio_path}: {e}")

    df = pd.DataFrame(records)
    df.to_csv(OUT_DIR / "asr_cleanup_output.csv", index=False)
    return df


if __name__ == "__main__":
    demo_number_normalisation()
    demo_english_detection()