Josh Talks — AI Researcher Intern (Speech & Audio)
Task Submission Report
Submitted by: Sumant Yadav
Email: sumantyadav3086@gmail.com
GitHub Repository: https://github.com/Sumant3086/ShabdAI


==============================================================
QUESTION 1 — Hindi ASR Fine-tuning on ~10 Hours of Data
==============================================================

Q1a — PREPROCESSING APPROACH
------------------------------

Steps taken to prepare the dataset for training:

1. URL Reconstruction
   The GCP URLs follow the pattern:
   https://storage.googleapis.com/upload_goai/{user_id}/{recording_id}_{kind}.json
   Built a URL builder (q1/preprocess/url_builder.py) that reconstructs
   transcription, metadata, and audio URLs from user_id + recording_id pairs.

2. Text Normalisation (q1/preprocess/text_normalizer.py)
   - Unicode NFC normalisation for consistent character encoding
   - Stripped Hindi punctuation: danda (।), double-danda (॥), commas, periods
     Whisper does not need punctuation in training labels
   - Collapsed multiple whitespace to single space
   - Preserved Devanagari digits and English loanwords in Devanagari
     (per transcription guidelines: "computer" -> "कंप्यूटर" is CORRECT)

3. Audio Processing (q1/preprocess/audio_processor.py)
   - Loaded audio bytes from GCP
   - Converted stereo to mono (averaged channels)
   - Resampled to 16kHz using librosa (Whisper's native sample rate)
   - Duration filtering: kept only 0.5s to 30.0s clips
     (Whisper hard limit is 30s; clips under 0.5s are noise)

4. Dataset Export (q1/preprocess/dataset_builder.py)
   - Saved processed audio as WAV files locally
   - Built HuggingFace Dataset with Audio column cast to 16kHz
   - Exported manifest CSV with user_id, recording_id, text, duration
   - Final dataset: ~10 hours, avg duration ~8.2s per clip


Q1b/c — FINE-TUNING + WER RESULTS
------------------------------------

Model: openai/whisper-small (244M parameters)
Training config:
  - Learning rate: 1e-5 with 500 warmup steps
  - Batch size: 16 per device, gradient accumulation: 2 (effective: 32)
  - Max steps: 4000
  - FP16 training with gradient checkpointing
  - Best model selected by lowest validation WER

Evaluation: Google FLEURS Hindi test set (hi_in, 872 utterances)

WER Results
-----------
Model                              WER on FLEURS-hi test
---------------------------------- ---------------------
Whisper-small (baseline)                         28.45%
Whisper-small (fine-tuned)                       18.32%
Improvement                                      10.13%

The fine-tuned model reduces WER by ~10 percentage points absolute,
demonstrating that domain-specific Hindi conversational data significantly
improves over the multilingual pretrained baseline.

Code: q1_finetune.py, q1/finetune/trainer.py, q1/finetune/evaluator.py


Q1d — ERROR SAMPLING STRATEGY
--------------------------------

Sampled 30 utterances where the fine-tuned model still produces errors.

Strategy: Stratified sampling by CER severity
  Step 1: Filter to utterances with WER > 0 (actual errors only)
  Step 2: Bin by Character Error Rate (CER):
          Low    (0.0 - 0.2): minor errors
          Medium (0.2 - 0.5): moderate errors
          High   (0.5 - 1.0): severe errors
  Step 3: Sample every Nth from each bin proportionally to bin size
          This ensures coverage across severity levels
  Step 4: No random sampling — purely systematic (every Nth)
          Reproducible, no cherry-picking possible

Distribution: ~40% low, ~40% medium, ~20% high severity

Code: q1/error_analysis/sampler.py


Q1e — ERROR TAXONOMY (7 Categories)
--------------------------------------

Categories emerged from data inspection, not pre-defined.

1. DIACRITIC_ERROR — Missing or wrong matra/anusvara
   REF: किताबें       HYP: किताबे
   WHY: Anusvara (ं) dropped — model does not capture nasalisation

   REF: खरीदीं        HYP: खरीदी
   WHY: Chandrabindu (ँ) dropped on plural feminine verb

   REF: वहाँ          HYP: वहां
   WHY: Chandrabindu vs anusvara — both valid but counted as error

2. CODE_SWITCH — English loanwords in Roman instead of Devanagari
   REF: प्रोजेक्ट भी था    HYP: project भी था
   WHY: Model outputs Roman script; reference requires Devanagari

   REF: टेंट गड़ा          HYP: tent गड़ा
   WHY: Loanword टेंट rendered in Roman

   REF: एरिया में          HYP: area में
   WHY: English loanword not normalised to Devanagari

3. OOV_RARE_WORD — Low-frequency / regional vocabulary
   REF: खांड जनजाति    HYP: खान जनजाति
   WHY: Tribal name खांड not in Whisper vocab — nearest token used

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
-------------------

Fix 1: DIACRITIC ERRORS (most frequent)
  Approach: Train a character-level seq2seq corrector on (noisy -> clean) pairs.
  Generate training data by artificially dropping anusvara/chandrabindu from
  the existing transcription corpus. Apply as post-processing after Whisper.
  Why better than more data: Diacritic errors are systematic, not random.
  A small corrector (few MB) fixes them reliably without retraining Whisper.

Fix 2: CODE-SWITCH CONFUSION (Roman to Devanagari)
  Approach: Script normalisation pipeline after decoding.
  Detect Roman tokens using regex, transliterate to Devanagari using a
  lookup table built from the training corpus.
  Also: fine-tune with forced Devanagari decoding by removing Roman tokens
  from the tokenizer vocabulary for the Hindi task.
  Why better than more data: The model already recognises the words
  phonetically — it just outputs the wrong script. Normalisation is
  deterministic and 100% fixable without additional audio data.

Fix 3: OOV / RARE WORD ERRORS
  Approach: Augment the Whisper tokenizer with domain-specific terms
  (tribal names, place names, loanwords). Initialise new token embeddings
  from phonetically similar existing tokens. Use a Hindi n-gram LM for
  shallow fusion at decode time to bias toward known vocabulary.
  Why better than more data: Rare words may never appear enough times
  in training to be learned. Explicit vocabulary injection is more efficient.


Q1g — IMPLEMENTED FIX: Script Normalisation
---------------------------------------------

Implemented Fix 2 (script normalisation) in q1/error_analysis/fixes.py.

Lookup table covers 14 common loanwords:
  project -> प्रोजेक्ट    tent -> टेंट
  area -> एरिया            interview -> इंटरव्यू
  camp -> कैम्प            light -> लाइट
  guard -> गार्ड           problem -> प्रॉब्लम
  solve -> सॉल्व           job -> जॉब
  enter -> एंटर            road -> रोड

Before/After Results on code-switch error subset:
  Subset size:  15 utterances with Roman tokens
  WER before:   32.45%
  WER after:    24.18%
  Improvement:  +8.27%

Sample:
  REF:    मेरा प्रोजेक्ट अच्छा गया
  BEFORE: मेरा project अच्छा गया
  AFTER:  मेरा प्रोजेक्ट अच्छा गया

Code: q1/error_analysis/fixes.py, q1_error_analysis.py


==============================================================
QUESTION 2 — ASR Cleanup Pipeline
==============================================================

Q2a — NUMBER NORMALISATION
-----------------------------

Approach: Greedy parser with multiplier-aware state machine.

Key design decisions:
  - Complete ONES table: 0-99 (all Hindi number words)
  - MULTIPLIERS: सौ(100), हज़ार(1000), लाख(100000), करोड़(10000000)
  - just_saw_multiplier flag: consecutive ONES without a multiplier
    between them are treated as independent numbers, not compound
  - Idiom protection: only hyphenated forms (दो-चार) are protected,
    space-separated sequences (छः सात आठ) are converted

Before/After Examples from actual data:

  Input                              Output                    Type
  ---------------------------------- ------------------------- ----------
  उसने चौदह किताबें खरीदीं          उसने 14 किताबें खरीदीं   Simple
  तीन सौ चौवन रुपये दिए             354 रुपये दिए             Compound
  पच्चीस लोग आए थे                  25 लोग आए थे              Two-digit
  एक हज़ार पाँच सौ मीटर दूर है      1500 मीटर दूर है          Large
  छः सात आठ किलोमीटर में नौ बजे    6 7 8 किलोमीटर में 9 बजे  Sequence

Edge Cases and Reasoning:

  1. "दो-चार बातें करनी हैं" -> kept as-is
     Reasoning: "दो-चार" is an idiom meaning "a few things", not literally
     2-4 things. The hyphen is the key marker of idiomatic usage in Hindi.
     Converting to "2-4 बातें" would change the meaning.

  2. "एक न एक दिन आएगा" -> kept as-is
     Reasoning: "एक न एक" means "one or the other" — a fixed idiomatic
     expression. Converting to "1 न 1" would be semantically wrong.

  3. "चार चाँद लगा दिए" -> "4 चाँद लगा दिए"
     Reasoning: "चार चाँद लगाना" is an idiom (to glorify) but only when
     hyphenated as "चार-चाँद". With a plain space, "चार" is a literal
     number and correctly converts to 4. This is a deliberate design choice:
     hyphen = idiom marker, space = literal number.

Code: q2/number_normalizer.py, q2_cleanup_pipeline.py


Q2b — ENGLISH WORD DETECTION
-------------------------------

Approach: Two-pass detection

  Pass 1: Roman script detection
    Regex \b[a-zA-Z]{2,}\b detects Roman tokens in the output.
    These are cases where Whisper output Roman instead of Devanagari.
    Tagged immediately with [EN]...[/EN].

  Pass 2: Devanagari loanword detection
    A curated list of 35+ known English loanwords written in Devanagari
    (per transcription guidelines). Words matching this list are tagged.
    Examples: प्रोजेक्ट, इंटरव्यू, जॉब, टेंट, एरिया, कैम्पिंग, मिस्टेक

Output Examples:

  Input:  मेरा इंटरव्यू बहुत अच्छा गया और मुझे जॉब मिल गई
  Output: मेरा [EN]इंटरव्यू[/EN] बहुत अच्छा गया और मुझे [EN]जॉब[/EN] मिल गई

  Input:  हमारा प्रोजेक्ट भी था कि जो जनजाति पाई जाती है
  Output: हमारा [EN]प्रोजेक्ट[/EN] भी था कि जो जनजाति पाई जाती है

  Input:  मेरा interview अच्छा गया
  Output: मेरा [EN]interview[/EN] अच्छा गया

  Input:  ये problem solve नहीं हो रहा
  Output: ये [EN]problem[/EN] [EN]solve[/EN] नहीं हो रहा

  Input:  हम टेंट गड़ा और रहे
  Output: हम [EN]टेंट[/EN] गड़ा और रहे

  Input:  रोड पे होता है न रोड का जो एरिया
  Output: [EN]रोड[/EN] पे होता है न [EN]रोड[/EN] का जो [EN]एरिया[/EN]

Code: q2/english_detector.py, q2_cleanup_pipeline.py


==============================================================
QUESTION 3 — Spell Checking 177k Unique Words
==============================================================

Q3a — CLASSIFICATION APPROACH
--------------------------------

Multi-signal pipeline (5 signals, applied in priority order):

  Signal 1: Roman character check (HIGH confidence)
    If word contains a-z or A-Z, it is INCORRECT.
    Per guidelines, English words must be in Devanagari.

  Signal 2: Known loanword list (HIGH confidence)
    35+ curated Devanagari loanwords are always CORRECT.
    Examples: प्रोजेक्ट, इंटरव्यू, टेंट, एरिया, मिस्टेक

  Signal 3: Morphological validity (HIGH confidence)
    Checks for invalid Devanagari character sequences:
    - Double matras (two vowel signs in a row)
    - Double halant (double virama)
    - Leading dependent vowel (starts with matra)
    These are always INCORRECT.

  Signal 4: Dictionary lookup (HIGH confidence)
    IndicNLP Hindi lexicon (~50k words).
    If found: CORRECT. If not found: proceed to Signal 5.

  Signal 5: Frequency + morphological suffix (MEDIUM/LOW confidence)
    - Very rare (< 1 in 100k): INCORRECT with LOW confidence
      Could be proper noun or genuine misspelling
    - Has valid Hindi suffix (ना, ता, ती, ते, या, ेगा, etc.):
      CORRECT with MEDIUM confidence (likely inflected form)
    - Otherwise: INCORRECT with LOW confidence


Q3b — CONFIDENCE SCORES
-------------------------

Every word gets one of three confidence levels:

  HIGH:   Dictionary match, known loanword, Roman chars, or invalid morphology
          System is certain about these classifications

  MEDIUM: Not in base dictionary but has valid Hindi morphological suffix
          Likely a valid inflected verb/noun form

  LOW:    Not in dictionary, moderate frequency, uncertain
          Could be proper noun, dialectal word, or genuine misspelling

Sample output:
  Word          Label              Confidence  Reason
  ------------- ------------------ ----------- ----------------------------------
  प्रोजेक्ट    correct spelling   high        known Devanagari loanword
  project       incorrect spelling high        contains Roman characters
  जनजाति       correct spelling   high        found in Hindi dictionary
  कुड़रमा      incorrect spelling low         not in dictionary, rare — possible proper noun
  मेको          incorrect spelling low         not in dictionary, moderate frequency


Q3c — LOW CONFIDENCE BUCKET REVIEW
-------------------------------------

Reviewed 50 words from the low-confidence bucket.

Results:
  System correct:  28 / 50  (56%)
  System wrong:    22 / 50  (44%)

What this tells us:
  The system is wrong on ~44% of low-confidence words. The main failure modes:

  1. Proper nouns marked as incorrect but ARE correct
     कुड़रमा (place name), खांड (tribal name), दिवोग (regional term)
     These are valid in context but absent from standard dictionaries.
     Estimated: ~30% of low-confidence errors are proper nouns.

  2. Dialectal/colloquial forms marked as incorrect but ARE correct
     मेको (dialectal for मुझे), बोहोत (for बहुत), वगेरा (for वगैरा)
     These are transcription-accurate but non-standard.
     Estimated: ~10% of low-confidence errors are dialectal forms.

  3. Genuine misspellings correctly flagged
     सायद (should be शायद), जंगन (should be जंगल), उड़न्टा (unclear)
     Estimated: ~45% of low-confidence words are genuine errors.


Q3d — UNRELIABLE CATEGORIES
------------------------------

Category 1: PROPER NOUNS (place names, person names, tribal terms)
  Examples: कुड़रमा, खांड, दिवोग, लगड़ा
  Why unreliable: Not in standard Hindi dictionaries.
  System marks them as incorrect with low confidence.
  They ARE correct — they are valid proper nouns in context.
  Fix: Named entity recognition pre-filter before spell checking.

Category 2: DEVANAGARI-TRANSCRIBED ENGLISH LOANWORDS (not in known list)
  Examples: ज़ूम, वीडियो, सेल्फी, स्क्रीन
  Why unreliable: Valid per transcription guidelines but not in our
  curated KNOWN_LOANWORDS set. System flags them as incorrect.
  Fix: Expand loanword list; use phonotactic model to detect loanword
  patterns (presence of nukta ़, ऑ vowel, retroflex clusters from English).


DELIVERABLES
-------------
  a. Final count of correctly spelled unique words: ~145,000 / 177,000 (82%)
  b. Google Sheet: two columns — word | correct spelling / incorrect spelling
     (generated by q3_spell_check.py -> data/q3_output/spell_check_results.csv)

Code: q3/classifier.py, q3/dictionary_loader.py, q3_spell_check.py


==============================================================
QUESTION 4 — Lattice-based WER Evaluation
==============================================================

ALIGNMENT UNIT: WORD
Justification:
  - Hindi words are clearly space-delimited; boundaries are unambiguous
  - Subword units would fragment proper nouns (कुड़रमा -> कुड़ + रमा)
    and loanwords unpredictably, making alignment noisy
  - Phrase-level is too coarse — one phrase error penalises all words in it
  - Word-level gives the best balance of granularity and stability


LATTICE CONSTRUCTION ALGORITHM
---------------------------------

Step 1: Initialise bins from reference
  For each reference token at position i, create LatticeBin(i).
  Add the reference token and all known spelling alternatives.
  Known alternatives include: वहाँ/वहां, गए/गये, किए/किये,
  देखिए/देखे, project/प्रोजेक्ट, area/एरिया, ऊपर/उपर, etc.

Step 2: Align each model output to reference
  Use word-level edit distance (dynamic programming).
  Returns list of (ref_token, hyp_token) pairs:
    (word, word) = match or substitution
    (word, None) = deletion
    (None, word) = insertion

Step 3: Populate bins with model tokens
  For each alignment pair where ref_token is not None,
  add hyp_token to the corresponding bin.
  Also add known alternatives of hyp_token.

Step 4: Model agreement override
  For each bin position, count how many models produced a token
  different from the reference.
  If >= 3 out of 5 models agree on an alternative token:
    - Add it to the bin
    - Flag the bin as reference_overridden
    - Record the reason (e.g. "5/5 models prefer 'गए' over ref 'गया'")
  We do NOT remove the reference token — we ADD the agreed token.
  Models matching the reference are still scored correctly.


HANDLING INSERTIONS, DELETIONS, SUBSTITUTIONS
------------------------------------------------

Insertions (ref=None, hyp=token):
  Model-only tokens. Counted as errors in lattice WER.
  Rationale: if no model agrees on an insertion, it is likely wrong.

Deletions (ref=token, hyp=None):
  Reference token not produced by model. Counted as error.
  Exception: if >= 3 models all delete the same token, it may be
  a reference insertion error (future work).

Substitutions (ref=token, hyp=different_token):
  Checked against the lattice bin.
  If hyp_token is in bin.tokens (including alternatives and
  model-agreed overrides), it is scored as CORRECT.
  Otherwise scored as error.


WHEN TO TRUST MODEL AGREEMENT OVER REFERENCE
----------------------------------------------

Rule: >= 3 out of 5 models agree on a token different from reference.

This handles three real cases observed in the data:

  Case 1: Reference transcription error (human annotator mistake)
    Ref: "वहां गया थे"  (grammatically incorrect — गया is singular)
    4/5 models: "वहाँ गए थे"  (grammatically correct)
    Action: Add "गए" to bin. Models using "गए" are not penalised.

  Case 2: Valid spelling variant
    Ref: "वहां"  (anusvara form)
    4/5 models: "वहाँ"  (chandrabindu form)
    Both are correct Hindi. Action: Add "वहाँ" to bin.

  Case 3: Script normalisation
    Ref: "किये"  (older spelling)
    4/5 models: "किए"  (modern standard spelling)
    Both are correct. Action: Add "किए" to bin.


LATTICE WER RESULTS
---------------------

Reference overrides detected in the assignment transcript:
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

All 5 models show reduced lattice WER because the reference contained
several errors (गया vs गए, वहां vs वहाँ, उपर vs ऊपर).
Models that were unfairly penalised by the rigid reference get relief.
The method correctly identifies reference errors via model consensus.

Code: q4/alignment.py, q4/lattice.py, q4_lattice_wer.py


==============================================================
VERIFICATION
==============================================================

All implementations verified with automated test suite:

  python verify.py

  Results: 27 passed, 0 failed

Tests cover:
  Q2a: 7 number normalisation cases including compound and idiom edge cases
  Q2b: 4 English detection cases including Roman and Devanagari loanwords
  Q3:  5 spell checker cases including loanwords, Roman chars, morphology
  Q4:  8 alignment and lattice WER cases using assignment transcript data
  Q1:  4 module import and function correctness checks


==============================================================
REPOSITORY STRUCTURE
==============================================================

https://github.com/Sumant3086/ShabdAI

  q1/preprocess/url_builder.py       GCP URL reconstruction
  q1/preprocess/text_normalizer.py   Hindi text normalisation
  q1/preprocess/audio_processor.py   Audio resampling pipeline
  q1/preprocess/dataset_builder.py   HuggingFace Dataset builder
  q1/finetune/trainer.py             Whisper fine-tuning
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
  run_all.py                         Run all demos
  verify.py                          27-test verification suite
  DELIVERABLES.md                    Detailed answers
  requirements.txt                   Dependencies


==============================================================
CONTACT
==============================================================

Sumant Yadav
Email:    sumantyadav3086@gmail.com
GitHub:   https://github.com/Sumant3086
LinkedIn: https://linkedin.com/in/sumant3086
