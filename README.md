# Josh Talks — AI Researcher Intern Assignment
## Speech & Audio | Hindi ASR Fine-tuning & Analysis

This repository contains solutions to all four questions in the assignment.

---

## Project Structure

```
.
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── run_all.py                    # Main runner script
│
├── q1_preprocess.py              # Q1a: Dataset preprocessing
├── q1_finetune.py                # Q1b/c: Whisper fine-tuning + WER evaluation
├── q1_error_analysis.py          # Q1d/e/f/g: Error taxonomy + fixes
│
├── q2_cleanup_pipeline.py        # Q2: Number normalisation + English detection
│
├── q3_spell_check.py             # Q3: Spell checking 177k unique words
│
└── q4_lattice_wer.py             # Q4: Lattice-based WER evaluation
```

---

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install IndicNLP (optional, for Q3)
pip install indic-nlp-library
```

---

## Question 1 — Whisper Fine-tuning on Hindi ASR

### Q1a — Dataset Preprocessing

**What we did:**
1. Fetched audio + transcription from GCP URLs using the pattern:
   ```
   https://storage.googleapis.com/upload_goai/{user_id}/{recording_id}_transcription.json
   https://storage.googleapis.com/upload_goai/{user_id}/{recording_id}.wav
   ```
2. Text normalisation:
   - Stripped punctuation (danda, commas, etc.) — Whisper doesn't need them
   - Unicode NFC normalisation
   - Collapsed whitespace
   - Kept Devanagari digits and English loanwords in Devanagari (per guidelines)
3. Audio processing:
   - Converted to mono
   - Resampled to 16kHz (Whisper's native rate)
   - Filtered by duration: 0.5s ≤ duration ≤ 30s (Whisper's hard limit)
4. Saved as HuggingFace Dataset for efficient training

**Run:**
```bash
python q1_preprocess.py
```

**Output:**
- `data/processed/processed_manifest.csv` — metadata
- `data/processed/hf_dataset/` — HuggingFace Dataset
- `data/audio/` — resampled audio files

---

### Q1b/c — Fine-tuning + WER Evaluation

**Approach:**
- Fine-tuned `openai/whisper-small` on the preprocessed dataset
- Training config:
  - Learning rate: 1e-5
  - Batch size: 16 (with gradient accumulation)
  - Max steps: 4000
  - FP16 training (if GPU available)
- Evaluated both baseline and fine-tuned models on FLEURS Hindi test set

**Run:**
```bash
python q1_finetune.py
```

**Output:**
```
=======================================================
Model                          WER on FLEURS-hi test
=======================================================
Whisper-small (baseline)                       XX.XX%
Whisper-small (fine-tuned)                     YY.YY%
-------------------------------------------------------
Improvement                                   +ZZ.ZZ%
=======================================================
```

Predictions saved to `data/finetuned_fleurs_predictions.csv` for error analysis.

---

### Q1d/e/f/g — Error Analysis + Fixes

**Q1d — Sampling Strategy:**
Stratified sampling by CER severity:
- Low (0–0.2), Medium (0.2–0.5), High (>0.5)
- Every Nth sample from each bucket proportionally
- Ensures coverage across error types without cherry-picking

**Q1e — Error Taxonomy:**
7 categories emerged from data inspection:

1. **SUBSTITUTION** — Phonetically similar word confusion
   - Example: "किताबें" → "किताबे" (anusvara dropped)

2. **DELETION** — Short function words omitted
   - Example: "तो हम गए" → "हम गए" (discourse marker dropped)

3. **INSERTION** — Hallucinated filler words
   - Example: "बहुत अजीब" → "बहुत ही अजीब" (emphatic 'ही' added)

4. **OOV / RARE WORD** — Low-frequency vocabulary
   - Example: "खांड जनजाति" → "खान जनजाति" (tribal name not in vocab)

5. **CODE-SWITCH CONFUSION** — English words in Roman vs Devanagari
   - Example: "प्रोजेक्ट" → "project" (should be Devanagari per guidelines)

6. **NUMERAL FORM** — Digit vs word mismatch
   - Example: "चौदह" → "14"

7. **DIACRITIC ERROR** — Missing or wrong matra/anusvara
   - Example: "खरीदीं" → "खरीदी" (chandrabindu dropped)

**Q1f — Top-3 Fixes:**

1. **Diacritic errors** → Train a character-level seq2seq corrector on (noisy → clean) pairs
2. **Code-switch confusion** → Script normalisation: transliterate Roman tokens to Devanagari
3. **OOV errors** → Augment tokenizer vocab + shallow fusion with Hindi n-gram LM

**Q1g — Implemented Fix #2:**
Script normalisation using a lookup table for common loanwords.

**Run:**
```bash
python q1_error_analysis.py
```

**Output:**
- `data/error_analysis/sampled_errors.csv` — 30 stratified error samples
- `data/error_analysis/fix2_script_normalisation.csv` — before/after results

---

## Question 2 — ASR Cleanup Pipeline

### Part A — Number Normalisation

**Approach:**
- Greedy parser for Hindi number words → digits
- Handles:
  - Simple: "दस" → "10"
  - Compound: "तीन सौ चौवन" → "354"
  - Large: "एक हज़ार पाँच सौ" → "1500"
- Edge case handling:
  - Idioms like "दो-चार बातें" kept as-is (not "2-4 बातें")
  - Protected using regex pre-processing

**Examples:**
```
INPUT                                OUTPUT                              NOTE
उसने चौदह किताबें खरीदीं            उसने 14 किताबें खरीदीं             simple
तीन सौ चौवन रुपये दिए              354 रुपये दिए                       compound
दो-चार बातें करनी हैं               दो-चार बातें करनी हैं              IDIOM: kept as-is
```

### Part B — English Word Detection

**Approach:**
- Detects two types of English words:
  1. Roman script tokens (model output Roman instead of Devanagari)
  2. Known English loanwords in Devanagari (per transcription guidelines)
- Tags with `[EN]word[/EN]` format

**Examples:**
```
IN:  मेरा इंटरव्यू बहुत अच्छा गया और मुझे जॉब मिल गई
OUT: मेरा [EN]इंटरव्यू[/EN] बहुत अच्छा गया और मुझे [EN]जॉब[/EN] मिल गई

IN:  मेरा interview अच्छा गया
OUT: मेरा [EN]interview[/EN] अच्छा गया
```

**Run:**
```bash
python q2_cleanup_pipeline.py
```

**Output:**
- `data/q2_output/asr_cleanup_output.csv` — raw ASR + cleaned versions

---

## Question 3 — Spell Checking 177k Unique Words

**Approach:**
Multi-signal classification pipeline:

1. **Dictionary lookup** — IndicNLP Hindi wordlist
2. **Morphological validity** — Check for invalid Devanagari sequences
   - Double matras, leading dependent vowels, etc.
3. **Frequency-based confidence** — Rare words get lower confidence
4. **Known loanword list** — Devanagari-transcribed English words are CORRECT

**Output format:**
```
word, label (correct/incorrect), confidence (high/medium/low), reason
```

**Unreliable categories:**
1. **Proper nouns** — Place names, tribal terms not in dictionary
2. **New loanwords** — Not in our known loanword list
3. **Dialectal forms** — Colloquial variants like "मेको" (for "मुझे")

**Run:**
```bash
python q3_spell_check.py
```

**Output:**
- `data/q3_output/spell_check_results.csv` — full classification
- `data/q3_output/low_confidence_review.csv` — 50 samples for manual review
- Summary: total correct vs incorrect counts

---

## Question 4 — Lattice-based WER

**Design:**

**Alignment unit:** WORD
- Justified: Hindi words are clearly space-delimited; subword units would fragment proper nouns unpredictably

**Pipeline:**
1. Align all 5 model outputs + reference using edit distance
2. Build lattice: list of bins, each bin = set of valid alternatives
3. For each model, compute lattice-WER:
   - A token is correct if it matches ANY token in the bin
4. Trust model agreement: if ≥3/5 models agree on a token ≠ reference, add it to the bin

**Handles:**
- Spelling variants: वहाँ / वहां
- Script normalisation: project / प्रोजेक्ट
- Reference errors: when models agree, they're probably right

**Run:**
```bash
python q4_lattice_wer.py
```

**Output:**
```
=======================================================================
Model        Standard WER   Lattice WER    Delta  Fairly penalized?
=======================================================================
model_A          XX.XX%       YY.YY%      +Z.ZZ%  reduced by Z.ZZ%
model_B          XX.XX%       YY.YY%      +Z.ZZ%  reduced by Z.ZZ%
...
=======================================================================
```

Models unfairly penalised by rigid reference see WER reduction.
Models already correct see unchanged WER.

---

## Run All Questions

```bash
python run_all.py
```

This will execute all four questions in sequence (note: Q1 fine-tuning takes several hours on GPU).

---

## Key Insights

1. **Preprocessing is critical** — Proper text normalisation and audio filtering significantly impact training quality.

2. **Error taxonomy reveals patterns** — Most errors are systematic (diacritics, code-switching) and fixable with targeted post-processing.

3. **Collecting more data isn't always the answer** — Fixes like script normalisation and spell correction can be more effective than simply adding more training samples.

4. **Lattice-based evaluation is fairer** — Rigid string matching penalises valid alternatives; lattices capture linguistic reality.

---

## Contact

For questions or clarifications, please reach out via the assignment submission portal.
