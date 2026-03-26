import pdfplumber, json, re, sys

PDF = "TrainingData.pdf"

with pdfplumber.open(PDF) as pdf:
    print(f"Pages: {len(pdf.pages)}")
    full_text = ""
    for i, page in enumerate(pdf.pages):
        text = page.extract_text() or ""
        full_text += text + "\n"
        if i < 3:
            print(f"\n--- Page {i+1} ---")
            print(text[:2000])

# Try to find URLs
urls = re.findall(r'https?://[^\s\)\"\']+', full_text)
print(f"\n\nFound {len(urls)} URLs:")
for u in urls[:30]:
    print(" ", u)

# Try to find JSON-like structures
json_blocks = re.findall(r'\{[^{}]{20,}\}', full_text)
print(f"\nFound {len(json_blocks)} JSON-like blocks")
for b in json_blocks[:5]:
    print(" ", b[:200])

# Save full text
with open("training_data_extracted.txt", "w", encoding="utf-8") as f:
    f.write(full_text)
print(f"\nFull text saved to training_data_extracted.txt ({len(full_text)} chars)")
