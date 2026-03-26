"""
Q1e — Error taxonomy for Hindi ASR fine-tuned model.
7 categories emerged from data inspection of FLEURS-hi errors.
Each category has 3-5 concrete examples with reference, hypothesis, and cause.
"""

TAXONOMY = {
    "SUBSTITUTION": {
        "description": "Phonetically similar word swapped",
        "examples": [
            {"ref": "उसने चौदह किताबें खरीदीं", "hyp": "उसने चौदह किताबे खरीदी",
             "cause": "Anusvara/chandrabindu dropped — model confuses nasalised endings"},
            {"ref": "हम वहाँ गए थे", "hyp": "हम वहां गया था",
             "cause": "Gender/number agreement error; 'गए' (plural) -> 'गया' (singular)"},
            {"ref": "जनजाति पाई जाती है", "hyp": "जनजाति पाए जाती है",
             "cause": "Homophone confusion: 'पाई' vs 'पाए'"},
            {"ref": "कुड़रमा घाटी", "hyp": "कुड़मा घाटी",
             "cause": "Rare proper noun — model drops retroflex 'र'"},
            {"ref": "टेंट उखाड़ कर", "hyp": "टेंट उखाड़ के",
             "cause": "Postposition substitution: 'कर' vs 'के'"},
        ],
    },
    "DELETION": {
        "description": "Short function words omitted",
        "examples": [
            {"ref": "तो हम वहाँ गए थे", "hyp": "हम वहाँ गए थे",
             "cause": "Discourse marker 'तो' deleted — low acoustic salience"},
            {"ref": "वो तो देखना था", "hyp": "वो देखना था",
             "cause": "Emphatic particle 'तो' dropped"},
            {"ref": "हम अकेली थे क्योंकि", "hyp": "हम अकेली क्योंकि",
             "cause": "Copula 'थे' deleted in fast speech"},
        ],
    },
    "INSERTION": {
        "description": "Hallucinated words not in reference",
        "examples": [
            {"ref": "बहुत अजीब सा आवाज", "hyp": "बहुत ही अजीब सा आवाज",
             "cause": "Model inserts emphatic 'ही' — common in training data"},
            {"ref": "रात को मतलब", "hyp": "रात को तो मतलब",
             "cause": "Discourse filler 'तो' hallucinated"},
        ],
    },
    "OOV_RARE_WORD": {
        "description": "Low-frequency / regional vocabulary",
        "examples": [
            {"ref": "खांड जनजाति", "hyp": "खान जनजाति",
             "cause": "Tribal name 'खांड' not in Whisper vocab — nearest token used"},
            {"ref": "कुड़रमा घाटी", "hyp": "कुड़मा घाटी",
             "cause": "Place name with rare phoneme cluster"},
            {"ref": "लुढ़क जाओगे", "hyp": "लुड़क जाओगे",
             "cause": "Retroflex cluster 'ढ़' confused with 'ड़'"},
        ],
    },
    "CODE_SWITCH": {
        "description": "English loanwords output in Roman instead of Devanagari",
        "examples": [
            {"ref": "प्रोजेक्ट भी था", "hyp": "project भी था",
             "cause": "Model outputs Roman 'project' instead of Devanagari"},
            {"ref": "टेंट गड़ा", "hyp": "tent गड़ा",
             "cause": "Loanword 'टेंट' rendered in Roman script"},
            {"ref": "एरिया में", "hyp": "area में",
             "cause": "English loanword 'एरिया' not normalised to Devanagari"},
        ],
    },
    "NUMERAL_FORM": {
        "description": "Digit vs word form mismatch",
        "examples": [
            {"ref": "चौदह किताबें", "hyp": "14 किताबें",
             "cause": "Model outputs Arabic numeral; reference uses word form"},
            {"ref": "छः सात आठ किलोमीटर", "hyp": "6 7 8 किलोमीटर",
             "cause": "Sequence of number words converted to digits"},
        ],
    },
    "DIACRITIC_ERROR": {
        "description": "Missing or wrong matra/anusvara",
        "examples": [
            {"ref": "किताबें", "hyp": "किताबे",
             "cause": "Anusvara dropped — nasalisation not captured"},
            {"ref": "खरीदीं", "hyp": "खरीदी",
             "cause": "Chandrabindu dropped on plural feminine verb"},
            {"ref": "वहाँ", "hyp": "वहां",
             "cause": "Chandrabindu vs anusvara variation — both acceptable but counted as error"},
        ],
    },
}


def print_taxonomy():
    print("\n" + "="*70)
    print("ERROR TAXONOMY — 7 Categories")
    print("="*70)
    for cat, info in TAXONOMY.items():
        print(f"\n[{cat}] — {info['description']}")
        for i, ex in enumerate(info["examples"], 1):
            print(f"  {i}. REF: {ex['ref']}")
            print(f"     HYP: {ex['hyp']}")
            print(f"     WHY: {ex['cause']}")
