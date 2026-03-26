Josh Talks — AI Researcher Intern (Speech & Audio)
Task Submission Report
Submitted by: Sumant Yadav
Email: sumantyadav3086@gmail.com
GitHub: https://github.com/Sumant3086/ShabdAI


DATASET
=======

Source: TrainingData.pdf (provided by Josh Talks)
Parsed 104 records from PDF using interleaved-digit URL decoding.
Verified 90 accessible records via HTTP — 17.78 hours of Hindi audio.
14 records returned HTTP 404 (inaccessible GCP paths).
Manifest saved: data/training_manifest_valid.csv

Sample records:
  folder_id  recording_id  duration_sec  segments
  967179     825780        443s          24
  967179     825727        443s          38
  639950     526266        522s          40
  608692     542785        581s          58
  642712     523045        587s          43


==============================================================
QUESTION 1 — Hindi ASR Fine-tuning
==============================================================

Q1a — PREPROCESSING
--------------------

Code: q1/preprocess/ (url_builder, text_normalizer, audio_processor, dataset_builder)

Steps:

1. URL Reconstruction
   GCP pattern: https://storage.googleapis.com/upload_goai/{folder_id}/{recording_id}_{kind}.json
   Decoded folder IDs from garbled PDF URLs using digit-interleaving pattern.
   90 of 104 URLs verified accessible.

2. Text Normalisation
   - Unicode NFC normalisation
   - Stripped danda (।), double-danda (॥), commas, periods
   - Collapsed whitespace
   - Preserved Devanagari loanwords (per guidelines: "computer" -> "कंप्यूटर")

3. Audio Processing
   - Mono conversion (averaged stereo channels)
   - Resampled to 16kHz (Whisper native rate) using librosa
   - Duration filter: 0.5s to 30.0s (Whisper hard limit)

4. HuggingFace Dataset Export
   - Audio column cast to 16kHz
   - Manifest CSV: folder_id, recording_id, text, duration
   - Final: ~17.78 hours, avg ~8.2s per segment


Q1b/c — FINE-TUNING + WER
--------------------------

Code: q1_finetune.py, q1/finetune/trainer.py, q1/finetune/evaluator.py

Model: openai/whisper-small (244M parameters)
Training:
  Learning rate:    1e-5 with 500 warmup steps
  Batch size:       16 per device, gradient accumulation 2 (effective 32)
  Max steps:        4000
  FP16:             enabled (gradient checkpointing)
  Best model:       selected by lowest validation WER

Evaluation: Google FLEURS Hindi test set (hi_in, 872 utterances)

WER Results
-----------
  Model                          WER on FLEURS-hi test
  ------------------------------ ---------------------
  Whisper-small (baseline)                      28.45%
  Whisper-small (fine-tuned)                    18.32%
  Improvement                                   10.13%

Fine-tuning on domain-specific Hindi conversational data reduces WER
by ~10 percentage points absolute over the multilingual baseline.


Q1d — ERROR SAMPLING STRATEGY
------------------------------

Code: q1/error_analysis/sampler.py

Sampled 30 utterances with systematic stratified approach:
  Step 1: Keep only utterances with WER > 0
  Step 2: Bin by CER — Low (0-0.2), Medium (0.2-0.5), High (>0.5)
  Step 3: Sample every Nth from each bin proportionally
  Step 4: No random sampling — purely systematic, reproducible

Distribution: ~40% low, ~40% medium, ~20% high severity


Q1e — ERROR TAXONOMY (7 Categories)
-------------------------------------

Code: q1/error_analysis/taxonomy.py

1. DIACRITIC_ERROR — Missing anusvara/chandrabindu
   REF: किताबें    HYP: किताबे
   WHY: Anusvara dropped — model does not capture nasalisation
   REF: खरीदीं     HYP: खरीदी
   WHY: Chandrabindu dropped on plural feminine verb
   REF: वहाँ       HYP: वहां
   WHY: Chandrabindu vs anusvara — both valid but counted as error

2. CODE_SWITCH — English loanwords in Roman instead of Devanagari
   REF: प्रोजेक्ट भी था    HYP: project भी था
   WHY: Model outputs Roman; reference requires Devanagari
   REF: टेंट गड़ा          HYP: tent गड़ा
   WHY: Loanword rendered in Roman script
   REF: एरिया में          HYP: area में
   WHY: Loanword not normalised to Devanagari

3. OOV_RARE_WORD — Low-frequency / regional vocabulary
   REF: खांड जनजाति    HYP: खान जनजाति
   WHY: Tribal name not in Whisper vocab — nearest token used
   REF: कुड़रमा घाटी   HYP: कुड़मा घाटी
   WHY: Place name with rare phoneme cluster
   REF: लुढ़क जाओगे   HYP: लुड़क जाओगे
   WHY: Retroflex cluster ढ़ confused with ड़

4. SUBSTITUTION — Phonetically similar word swapped
   REF: जनजाति पाई जाती है    HYP: जनजाति पाए जाती है
   WHY: Homophone confusion: पाई vs पाए
   REF: हम वहाँ गए थे         HYP: हम वहां गया था
   WHY: Gender/number agreement error
   REF: टेंट उखाड़ कर         HYP: टेंट उखाड़ के
   WHY: Postposition substitution: कर vs के

5. DELETION — Short function words omitted
   REF: तो हम वहाँ गए थे    HYP: हम वहाँ गए थे
   WHY: Discourse marker तो deleted — low acoustic salience
   REF: वो तो देखना था       HYP: वो देखना था
   WHY: Emphatic particle तो dropped
   REF: हम अकेली थे क्योंकि  HYP: हम अकेली क्योंकि
   WHY: Copula थे deleted in fast speech

6. INSERTION — Hallucinated filler words
   REF: बहुत अजीब सा आवाज    HYP: बहुत ही अजीब सा आवाज
   WHY: Emphatic ही inserted — common in training data
   REF: रात को मतलब          HYP: रात को तो मतलब
   WHY: Discourse filler तो hallucinated

7. NUMERAL_FORM — Digit vs word form mismatch
   REF: चौदह किताबें          HYP: 14 किताबें
   WHY: Model outputs Arabic numeral; reference uses word form
   REF: छः सात आठ किलोमीटर   HYP: 6 7 8 किलोमीटर
   WHY: Sequence of number words converted to digits


Q1f — TOP-3 FIXES
------------------

Fix 1: DIACRITIC ERRORS (most frequent)
  Train a character-level seq2seq corrector on (noisy -> clean) pairs.
  Generate training data by artificially dropping anusvara/chandrabindu.
  Apply as post-processing after Whisper — no retraining needed.
  Why better than more data: errors are systematic, not random.

Fix 2: CODE-SWITCH CONFUSION
  Script normalisation: detect Roman tokens, transliterate to Devanagari
  using a lookup table built from the training corpus.
  Also: remove Roman tokens from tokenizer vocab for Hindi task.
  Why better than more data: model already recognises words phonetically —
  it just outputs the wrong script. Normalisation is 100% deterministic.

Fix 3: OOV / RARE WORD ERRORS
  Augment Whisper tokenizer with domain-specific terms.
  Initialise new embeddings from phonetically similar existing tokens.
  Use Hindi n-gram LM for shallow fusion at decode time.
  Why better than more data: rare words may never appear enough times
  in training. Explicit vocabulary injection is more efficient.


Q1g — IMPLEMENTED FIX: Script Normalisation
--------------------------------------------

Code: q1/error_analysis/fixes.py

Lookup table (14 loanwords):
  project -> प्रोजेक्ट    tent -> टेंट
  area -> एरिया            interview -> इंटरव्यू
  camp -> कैम्प            light -> लाइट
  guard -> गार्ड           problem -> प्रॉब्लम
  solve -> सॉल्व           job -> जॉब
  enter -> एंटर            road -> रोड

Before/After on code-switch error subset (15 utterances):
  WER before:   32.45%
  WER after:    24.18%
  Improvement:  +8.27%

Sample:
  REF:    मेरा प्रोजेक्ट अच्छा गया
  BEFORE: मेरा project अच्छा गया
  AFTER:  मेरा प्रोजेक्ट अच्छा गया


==============================================================
QUESTION 2 — ASR Cleanup Pipeline
==============================================================

Code: q2/number_normalizer.py, q2/english_detector.py, q2_cleanup_pipeline.py


Q2a — NUMBER NORMALISATION
---------------------------

Approach: Greedy parser with multiplier-aware state machine.

Design decisions:
  - Complete ONES table: 0-99 (all Hindi number words)
  - MULTIPLIERS: सौ(100), हज़ार(1000), लाख(100000), करोड़(10000000)
  - just_saw_multiplier flag: consecutive ONES without multiplier
    between them = independent numbers, not compound
  - Idiom protection: only hyphenated forms (दो-चार) protected;
    space-separated sequences (छः सात आठ) are converted

Before/After Examples:
  Input                              Output                    Type
  ---------------------------------- ------------------------- ----------
  उसने चौदह किताबें खरीदीं          उसने 14 किताबें खरीदीं   Simple
  तीन सौ चौवन रुपये दिए             354 रुपये दिए             Compound
  पच्चीस लोग आए थे                  25 लोग आए थे              Two-digit
  एक हज़ार पाँच सौ मीटर दूर है      1500 मीटर दूर है          Large
  छः सात आठ किलोमीटर में नौ बजे    6 7 8 किलोमीटर में 9 बजे  Sequence

Edge Cases:
  "दो-चार बातें करनी हैं" -> kept as-is
    Reasoning: दो-चार is an idiom meaning "a few things".
    Hyphen = idiom marker. Converting to "2-4 बातें" changes meaning.

  "एक न एक दिन आएगा" -> kept as-is
    Reasoning: "एक न एक" = "one or the other" — fixed idiomatic expression.

  "चार चाँद लगा दिए" -> "4 चाँद लगा दिए"
    Reasoning: "चार-चाँद" (hyphenated) is an idiom but "चार चाँद"
    (space-separated) is a literal number. Deliberate design choice.


Q2b — ENGLISH WORD DETECTION
------------------------------

Approach: Two-pass detection

  Pass 1: Roman script detection
    Regex detects Roman tokens — cases where Whisper output Roman
    instead of Devanagari. Tagged with [EN]...[/EN].

  Pass 2: Devanagari loanword detection
    35+ curated English loanwords in Devanagari are tagged.
    Per guidelines: English spoken words -> Devanagari is CORRECT.

Output Examples:
  IN:  मेरा इंटरव्यू बहुत अच्छा गया और मुझे जॉब मिल गई
  OUT: मेरा [EN]इंटरव्यू[/EN] बहुत अच्छा गया और मुझे [EN]जॉब[/EN] मिल गई

  IN:  हमारा प्रोजेक्ट भी था कि जो जनजाति पाई जाती है
  OUT: हमारा [EN]प्रोजेक्ट[/EN] भी था कि जो जनजाति पाई जाती है

  IN:  मेरा interview अच्छा गया
  OUT: मेरा [EN]interview[/EN] अच्छा गया

  IN:  ये problem solve नहीं हो रहा
  OUT: ये [EN]problem[/EN] [EN]solve[/EN] नहीं हो रहा

  IN:  हम टेंट गड़ा और रहे
  OUT: हम [EN]टेंट[/EN] गड़ा और रहे

  IN:  रोड पे होता है न रोड का जो एरिया
  OUT: [EN]रोड[/EN] पे होता है न [EN]रोड[/EN] का जो [EN]एरिया[/EN]


==============================================================
QUESTION 3 — Spell Checking 177k Unique Words
==============================================================

Code: q3/classifier.py, q3/dictionary_loader.py, q3_spell_check.py


Q3a — CLASSIFICATION APPROACH
-------------------------------

Multi-signal pipeline (5 signals, priority order):

  Signal 1: Roman character check (HIGH confidence)
    Contains a-z or A-Z -> INCORRECT
    Per guidelines, English words must be in Devanagari.

  Signal 2: Known loanword list (HIGH confidence)
    35+ curated Devanagari loanwords -> always CORRECT
    Examples: प्रोजेक्ट, इंटरव्यू, टेंट, एरिया, मिस्टेक

  Signal 3: Morphological validity (HIGH confidence)
    Invalid sequences -> INCORRECT:
    - Double matras (two vowel signs in a row)
    - Double halant (double virama)
    - Leading dependent vowel (starts with matra)

  Signal 4: Dictionary lookup (HIGH confidence)
    IndicNLP Hindi lexicon (~50k words)
    Found -> CORRECT. Not found -> Signal 5.

  Signal 5: Frequency + morphological suffix (MEDIUM/LOW)
    Very rare (< 1 in 100k) -> INCORRECT, LOW confidence
    Has valid Hindi suffix (ना, ता, ती, ते, ेगा...) -> CORRECT, MEDIUM
    Otherwise -> INCORRECT, LOW confidence


Q3b — CONFIDENCE SCORES
------------------------

  HIGH:   Dictionary match, known loanword, Roman chars, invalid morphology
  MEDIUM: Not in base dictionary but has valid Hindi morphological suffix
  LOW:    Not in dictionary, moderate frequency, uncertain

Sample:
  Word          Label              Confidence  Reason
  प्रोजेक्ट    correct spelling   high        known Devanagari loanword
  project       incorrect spelling high        contains Roman characters
  जनजाति       correct spelling   high        found in Hindi dictionary
  कुड़रमा      incorrect spelling low         not in dictionary, rare
  मेको          incorrect spelling low         not in dictionary, uncertain


Q3c — LOW CONFIDENCE REVIEW
-----------------------------

Reviewed 50 words from low-confidence bucket:
  System correct:  28 / 50  (56%)
  System wrong:    22 / 50  (44%)

Failure modes:
  ~30% proper nouns (कुड़रमा, खांड, दिवोग) — valid but not in dictionary
  ~10% dialectal forms (मेको, बोहोत) — transcription-accurate, non-standard
  ~45% genuine misspellings correctly flagged (सायद, जंगन)
  ~15% rare valid inflected forms incorrectly flagged


Q3d — UNRELIABLE CATEGORIES
-----------------------------

1. PROPER NOUNS — place names, tribal terms, person names
   Not in standard dictionaries. System marks as incorrect.
   Fix: Named entity recognition pre-filter.

2. DEVANAGARI LOANWORDS NOT IN KNOWN LIST
   Valid per guidelines but not in curated set.
   Fix: Expand list; use phonotactic model (nukta, ऑ vowel patterns).


DELIVERABLES
------------
  Correct words: ~145,000 / 177,000 (82%)
  Output file:   data/q3_output/spell_check_results.csv
  Columns:       word | label (correct/incorrect) | confidence | reason


==============================================================
QUESTION 4 — Lattice-based WER
==============================================================

Code: q4/alignment.py, q4/lattice.py, q4_lattice_wer.py


ALIGNMENT UNIT: WORD
Justification:
  Hindi words are clearly space-delimited — boundaries are unambiguous.
  Subword units fragment proper nouns (कुड़रमा -> कुड़ + रमा).
  Phrase-level is too coarse — one error penalises all words in phrase.
  Word-level gives best balance of granularity and stability.


LATTICE CONSTRUCTION
---------------------

Step 1: Initialise bins from reference
  For each reference token at position i, create LatticeBin(i).
  Add reference token + all known spelling alternatives.
  Known alternatives: वहाँ/वहां, गए/गये, किए/किये, देखिए/देखे,
  project/प्रोजेक्ट, area/एरिया, ऊपर/उपर, etc.

Step 2: Align each model output to reference
  Word-level edit distance (dynamic programming).
  Returns (ref_token, hyp_token) pairs:
    (word, word) = match or substitution
    (word, None) = deletion
    (None, word) = insertion

Step 3: Populate bins with model tokens
  Add hyp_token and its known alternatives to the bin.

Step 4: Model agreement override
  If >= 3/5 models agree on a token different from reference:
  - Add it to the bin (do NOT remove reference token)
  - Flag bin as reference_overridden with reason


HANDLING INSERTIONS, DELETIONS, SUBSTITUTIONS
----------------------------------------------

Insertions (ref=None, hyp=token): counted as errors
Deletions (ref=token, hyp=None): counted as errors
Substitutions: checked against lattice bin — if hyp_token is in
  bin.tokens (including alternatives + model-agreed overrides),
  scored as CORRECT. Otherwise scored as error.


WHEN TO TRUST MODEL AGREEMENT
-------------------------------

Rule: >= 3 out of 5 models agree on a token different from reference.

Case 1: Reference transcription error
  Ref: "वहां गया थे" (grammatically wrong — गया is singular)
  5/5 models: "वहाँ गए थे" (grammatically correct)
  Action: Add "गए" to bin. Models using "गए" not penalised.

Case 2: Valid spelling variant
  Ref: "वहां" (anusvara)  vs  4/5 models: "वहाँ" (chandrabindu)
  Both correct Hindi. Action: Add "वहाँ" to bin.

Case 3: Script normalisation
  Ref: "किये" (older)  vs  4/5 models: "किए" (modern standard)
  Both correct. Action: Add "किए" to bin.


LATTICE WER RESULTS
--------------------

Reference overrides detected in assignment transcript:
  Bin 9:  4/5 models prefer 'देखे' over ref 'देखिए'
  Bin 16: 4/5 models prefer 'वहाँ' over ref 'वहां'
  Bin 17: 5/5 models prefer 'गए' over ref 'गया'
  Bin 21: 4/5 models prefer 'किए' over ref 'किये'
  Bin 27: 5/5 models prefer 'गए' over ref 'गया'
  Bin 31: 5/5 models prefer 'ऊपर' over ref 'उपर'
  Bin 35: 4/5 models prefer 'वहाँ' over ref 'वहां'

WER Comparison:
  Model     Standard WER    Lattice WER    Reduction
  --------- --------------- -------------- ----------
  model_A   8.41%           0.00%          -8.41%
  model_B   8.23%           0.65%          -7.58%
  model_C   16.80%          8.85%          -7.95%
  model_D   5.44%           0.00%          -5.44%
  model_E   9.07%           2.79%          -6.28%

All 5 models show reduced lattice WER. Models unfairly penalised by
reference errors get relief. Exact-match models stay at 0%.


==============================================================
VERIFICATION
==============================================================

python verify.py  ->  Results: 27 passed, 0 failed

Tests cover:
  Q2a: 7 number normalisation cases (simple, compound, idiom edge cases)
  Q2b: 4 English detection cases (Roman + Devanagari loanwords)
  Q3:  5 spell checker cases (loanwords, Roman chars, morphology)
  Q4:  8 alignment + lattice WER cases using assignment transcript
  Q1:  4 module import + function correctness checks


==============================================================
PROJECT STRUCTURE
==============================================================

https://github.com/Sumant3086/ShabdAI

  q1/preprocess/url_builder.py       GCP URL reconstruction
  q1/preprocess/text_normalizer.py   Hindi text normalisation
  q1/preprocess/audio_processor.py   Audio resampling pipeline
  q1/preprocess/dataset_builder.py   HuggingFace Dataset builder
  q1/finetune/trainer.py             Whisper fine-tuning config
  q1/finetune/evaluator.py           FLEURS WER evaluation
  q1/error_analysis/sampler.py       Stratified error sampling
  q1/error_analysis/taxonomy.py      7-category error taxonomy
  q1/error_analysis/fixes.py         Script normalisation fix
  q2/number_normalizer.py            Hindi number to digit parser
  q2/english_detector.py             English word tagger
  q3/dictionary_loader.py            Hindi dictionary loader
  q3/classifier.py                   Multi-signal spell checker
  q4/alignment.py                    Edit distance alignment
  q4/lattice.py                      Lattice builder + WER
  q1_preprocess.py                   Q1a entry point
  q1_finetune.py                     Q1b/c entry point
  q1_error_analysis.py               Q1d-g entry point
  q2_cleanup_pipeline.py             Q2 entry point
  q3_spell_check.py                  Q3 entry point
  q4_lattice_wer.py                  Q4 entry point
  run_all.py                         Run all demos
  verify.py                          27-test verification suite
  requirements.txt                   Python dependencies
  data/training_manifest_valid.csv   90 verified GCP records


==============================================================
CONTACT
==============================================================

Sumant Yadav
Email:    sumantyadav3086@gmail.com
GitHub:   https://github.com/Sumant3086
LinkedIn: https://linkedin.com/in/sumant3086
