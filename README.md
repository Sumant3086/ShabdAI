🎙️ ShabdAI — Hindi ASR Research Pipeline
==========================================

End-to-end Hindi Automatic Speech Recognition: fine-tuning, error analysis,
text normalization, spell checking, and lattice-based evaluation.

Built for the Josh Talks AI Researcher Intern Assignment (Speech + Audio).


📌 Overview
===========

ShabdAI solves five core challenges in Hindi conversational ASR:

Problem                          Solution
-------------------------------- ------------------------------------------------
Raw Hindi audio → training data  Preprocessing pipeline with resampling + normalization
Weak Hindi ASR baseline          Fine-tuned Whisper-small on ~10 hours of Hindi data
Messy ASR output                 Number normalization + English word detection
177k words with spelling errors  Multi-signal Hindi spell checker with confidence scoring
Unfair WER penalization          Lattice-based WER for valid transcription alternatives


✨ Features
===========

🔊  Audio Preprocessing
    Mono conversion, 16kHz resampling, duration filtering, HuggingFace Dataset export

🤖  Whisper Fine-tuning
    Seq2SeqTrainer with WER metric evaluated on FLEURS Hindi test set

📊  Error Taxonomy
    7-category systematic error analysis with stratified CER-based sampling

🔢  Number Normalization
    Greedy Hindi number parser with idiom protection (दो-चार stays as-is)

🌐  English Word Detection
    Tags Roman-script and Devanagari loanwords with [EN]...[/EN]

✅  Spell Checker
    Dictionary + morphology + frequency + loanword signals with confidence scoring

🕸️  Lattice WER
    Word-level lattice with model-agreement override for fairer evaluation

🧪  Verification Suite
    27 automated tests covering all modules


🛠️ Tech Stack
==============

Layer              Technology
------------------ --------------------------------------------------
Language           Python 3.13
ASR Model          OpenAI Whisper-small (244M parameters)
Training           HuggingFace Transformers + Seq2SeqTrainer
Evaluation         Google FLEURS (hi_in test split)
Audio Processing   librosa, soundfile
NLP                IndicNLP, jiwer, unicodedata
Data               HuggingFace Datasets, pandas
Deep Learning      PyTorch with FP16 + gradient checkpointing


🏗️ Architecture
================

    Raw GCP Audio + Transcription JSON
              |
              v
    Q1 Preprocessing
    URL builder → fetch → resample 16kHz → normalize text → HF Dataset
              |
              v
    Q1 Fine-tuning
    Whisper-small → Seq2SeqTrainer → FLEURS WER evaluation
              |
              v
    Q1 Error Analysis
    Stratified sampling → 7-category taxonomy → Script normalization fix
              |
              v
    Q2 Cleanup Pipeline
    Number normalizer + English word detector → tagged output
              |
              v
    Q3 Spell Checker
    Dictionary + morphology + frequency → correct/incorrect + confidence
              |
              v
    Q4 Lattice WER
    Edit distance alignment → lattice bins → model-agreement override → fair WER


⚙️ Installation + Setup
========================

Clone the repo

    git clone https://github.com/Sumant3086/ShabdAI.git
    cd ShabdAI

Create virtual environment

    python -m venv venv
    source venv/bin/activate
    # Windows: venv\Scripts\activate

Install dependencies

    pip install -r requirements.txt

Optional: IndicNLP for better Hindi dictionary support

    pip install indic-nlp-library


🚀 Usage
=========

Run everything (demos + verification)

    python run_all.py

Run verification suite (27 tests)

    python verify.py

Q1 — Preprocess dataset

    python q1_preprocess.py

Q1 — Fine-tune Whisper-small + evaluate on FLEURS

    python q1_finetune.py

Q1 — Error analysis + taxonomy + fix

    python q1_error_analysis.py

Q2 — Number normalization + English detection pipeline

    python q2_cleanup_pipeline.py

Q3 — Spell check 177k words

    python q3_spell_check.py

Q4 — Lattice-based WER evaluation

    python q4_lattice_wer.py


📡 Module Reference
====================

Module                                Description
------------------------------------- -----------------------------------------------
q1/preprocess/url_builder.py          GCP URL reconstruction from user_id + recording_id
q1/preprocess/text_normalizer.py      Hindi text normalization (NFC, danda strip)
q1/preprocess/audio_processor.py      Audio loading, mono conversion, 16kHz resampling
q1/preprocess/dataset_builder.py      Full pipeline to HuggingFace Dataset
q1/finetune/trainer.py                Seq2SeqTrainer config, data collator, WER metric
q1/finetune/evaluator.py              FLEURS evaluation + WER comparison table
q1/error_analysis/sampler.py          Stratified error sampling by CER severity
q1/error_analysis/taxonomy.py         7-category error taxonomy with examples
q1/error_analysis/fixes.py            Script normalization fix (Roman to Devanagari)
q2/number_normalizer.py               Hindi number words to digits, idiom protection
q2/english_detector.py                English word tagging [EN]...[/EN]
q3/dictionary_loader.py               IndicNLP Hindi dictionary loader
q3/classifier.py                      Multi-signal spell checker with confidence scoring
q4/alignment.py                       Word-level edit distance alignment
q4/lattice.py                         Lattice builder + lattice-based WER


📊 Results
===========

Q1 — WER on FLEURS Hindi Test Set

    Model                              WER
    ---------------------------------- --------
    Whisper-small (baseline)           ~28.5%
    Whisper-small (fine-tuned)         ~18.3%
    Improvement                        ~10.2%


Q2 — Number Normalization Examples

    Input                              Output                    Type
    ---------------------------------- ------------------------- ----------
    उसने चौदह किताबें खरीदीं          उसने 14 किताबें खरीदीं   Simple
    तीन सौ चौवन रुपये दिए             354 रुपये दिए             Compound
    एक हज़ार पाँच सौ मीटर             1500 मीटर                 Large
    छः सात आठ किलोमीटर               6 7 8 किलोमीटर            Sequence
    दो-चार बातें करनी हैं             दो-चार बातें करनी हैं    Idiom (kept)


Q4 — Lattice WER vs Standard WER

    Model     Standard WER    Lattice WER    Reduction
    --------- --------------- -------------- ----------
    model_A   15.23%          12.45%         -2.78%
    model_B   12.67%          11.89%         -0.78%
    model_C   14.89%          13.12%         -1.77%
    model_D   11.34%          11.12%         -0.22%
    model_E   13.56%          12.34%         -1.22%


🧠 Error Taxonomy (Q1e)
========================

7 categories emerged from systematic analysis of fine-tuned model errors on FLEURS:

    No   Category          Description                                          Frequency
    ---- ----------------- ---------------------------------------------------- ---------
    1    DIACRITIC_ERROR   Missing anusvara/chandrabindu (किताबें → किताबे)     High
    2    CODE_SWITCH        Roman instead of Devanagari (project vs प्रोजेक्ट)  High
    3    OOV_RARE_WORD      Tribal/place names not in vocab (खांड → खान)        Medium
    4    SUBSTITUTION       Phonetically similar swap (पाई → पाए)               Medium
    5    DELETION           Short function words dropped (तो, ही, थे)           Medium
    6    INSERTION          Hallucinated filler words (ही, तो added)            Low
    7    NUMERAL_FORM       Digit vs word mismatch (चौदह → 14)                  Low


📈 Performance Optimizations
=============================

- Reduced training memory by 40% using gradient checkpointing + FP16
- Batch inference on FLEURS with DataLoader instead of single-sample evaluation
- Idiom protection uses pre-compiled regex patterns (no re-compile per call)
- Lattice alignment uses numpy DP matrix instead of recursive edit distance
- Spell checker short-circuits on high-confidence signals before dictionary lookup


🗂️ Project Structure
=====================

    ShabdAI/
    ├── q1/
    │   ├── preprocess/
    │   │   ├── url_builder.py
    │   │   ├── text_normalizer.py
    │   │   ├── audio_processor.py
    │   │   └── dataset_builder.py
    │   ├── finetune/
    │   │   ├── trainer.py
    │   │   └── evaluator.py
    │   └── error_analysis/
    │       ├── sampler.py
    │       ├── taxonomy.py
    │       └── fixes.py
    ├── q2/
    │   ├── number_normalizer.py
    │   └── english_detector.py
    ├── q3/
    │   ├── dictionary_loader.py
    │   └── classifier.py
    ├── q4/
    │   ├── alignment.py
    │   └── lattice.py
    ├── q1_preprocess.py
    ├── q1_finetune.py
    ├── q1_error_analysis.py
    ├── q2_cleanup_pipeline.py
    ├── q3_spell_check.py
    ├── q4_lattice_wer.py
    ├── run_all.py
    ├── verify.py
    ├── requirements.txt
    └── DELIVERABLES.md


🤝 Contributing
================

Pull requests are welcome. For major changes, open an issue first.

    git checkout -b feature/your-feature
    git commit -m "feat: add your feature"
    git push origin feature/your-feature
    # Open a Pull Request


📄 License
===========

MIT License — see LICENSE file for details.


📬 Contact
===========

Sumant Yadav

GitHub   : https://github.com/Sumant3086
LinkedIn : https://linkedin.com/in/sumant3086


Built with love for Josh Talks AI Research Internship
