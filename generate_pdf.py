"""
Generate submission PDF with proper Devanagari (Hindi) text rendering.
Uses Nirmala UI extracted from Windows TTC font collection.
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os, re

OUTPUT = "ShabdAI_Submission_SumantYadav.pdf"
REPORT = "SUBMISSION_REPORT.md"

# Register Nirmala UI (supports Devanagari)
pdfmetrics.registerFont(TTFont("Nirmala", "Nirmala_0.ttf"))
pdfmetrics.registerFont(TTFont("NirmalaBold", "Nirmala_1.ttf"))
print("Font registered: Nirmala UI (Devanagari support)")

doc = SimpleDocTemplate(
    OUTPUT, pagesize=A4,
    rightMargin=18*mm, leftMargin=18*mm,
    topMargin=18*mm, bottomMargin=18*mm,
)

# Styles
cover_style = ParagraphStyle("Cover", fontName="NirmalaBold", fontSize=18,
    spaceAfter=4, textColor=colors.HexColor("#1a1a2e"), alignment=1, leading=24)
sub_style = ParagraphStyle("Sub", fontName="Nirmala", fontSize=10,
    spaceAfter=2, textColor=colors.HexColor("#444444"), alignment=1)
q_style = ParagraphStyle("Q", fontName="NirmalaBold", fontSize=13,
    spaceAfter=4, spaceBefore=10, textColor=colors.HexColor("#0f3460"), leading=18)
h2_style = ParagraphStyle("H2", fontName="NirmalaBold", fontSize=10,
    spaceAfter=3, spaceBefore=6, textColor=colors.HexColor("#16213e"), leading=14)
body_style = ParagraphStyle("Body", fontName="Nirmala", fontSize=9,
    spaceAfter=2, leading=14, textColor=colors.HexColor("#2d2d2d"))
indent_style = ParagraphStyle("Indent", fontName="Nirmala", fontSize=9,
    spaceAfter=1, leading=13, leftIndent=12, textColor=colors.HexColor("#333333"))
code_style = ParagraphStyle("Code", fontName="Courier", fontSize=8,
    spaceAfter=1, leading=11, leftIndent=12,
    textColor=colors.HexColor("#1a1a1a"), backColor=colors.HexColor("#f4f4f4"))

def safe(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

story = []

with open(REPORT, encoding="utf-8") as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    line = lines[i].rstrip("\n")

    # Cover lines
    if i < 6 and line.strip():
        if i == 0:
            story.append(Spacer(1, 8*mm))
            story.append(Paragraph(safe(line), cover_style))
        elif i == 1:
            story.append(Paragraph(safe(line), cover_style))
        else:
            story.append(Paragraph(safe(line), sub_style))
        i += 1
        continue

    # Section dividers
    if re.match(r"^={4,}", line):
        story.append(Spacer(1, 2*mm))
        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#0f3460")))
        story.append(Spacer(1, 1*mm))
        i += 1
        continue

    if re.match(r"^-{4,}", line):
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#bbbbbb")))
        i += 1
        continue

    # Question headers
    if re.match(r"^QUESTION \d", line):
        story.append(Spacer(1, 4*mm))
        story.append(Paragraph(safe(line), q_style))
        i += 1
        continue

    # Sub-headers
    if re.match(r"^Q\d[a-g]? —", line) or re.match(r"^[A-Z][A-Z0-9 /]+$", line.strip()) and len(line.strip()) > 3:
        story.append(Paragraph(safe(line), h2_style))
        i += 1
        continue

    # Blank line
    if line.strip() == "":
        story.append(Spacer(1, 2*mm))
        i += 1
        continue

    # Indented lines (code/table/examples)
    if line.startswith("  ") or line.startswith("\t"):
        # Check if it looks like a table row (has multiple spaces as columns)
        stripped = line.strip()
        if re.match(r'^[a-zA-Z\u0900-\u097F].*\s{3,}', stripped):
            story.append(Paragraph(safe(stripped), code_style))
        else:
            story.append(Paragraph(safe(stripped), indent_style))
        i += 1
        continue

    # Normal body text
    story.append(Paragraph(safe(line), body_style))
    i += 1

doc.build(story)
size_kb = os.path.getsize(OUTPUT) / 1024
print(f"\nPDF created: {OUTPUT}  ({size_kb:.1f} KB)")
print("Hindi text should now render correctly with Nirmala UI font.")
