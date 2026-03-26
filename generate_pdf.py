"""Generate submission PDF from SUBMISSION_REPORT.md"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os, re

OUTPUT = "ShabdAI_Submission_SumantYadav.pdf"
REPORT = "SUBMISSION_REPORT.md"

# Try to register a Unicode font for Hindi text
FONT_NAME = "Helvetica"  # fallback
for font_path in [
    r"C:\Windows\Fonts\Arial.ttf",
    r"C:\Windows\Fonts\NirmalaUI.ttf",
    r"C:\Windows\Fonts\mangal.ttf",
]:
    if os.path.exists(font_path):
        try:
            fname = os.path.splitext(os.path.basename(font_path))[0].replace(" ", "")
            pdfmetrics.registerFont(TTFont(fname, font_path))
            FONT_NAME = fname
            print(f"Using font: {font_path}")
            break
        except Exception:
            continue

doc = SimpleDocTemplate(
    OUTPUT,
    pagesize=A4,
    rightMargin=20*mm, leftMargin=20*mm,
    topMargin=20*mm, bottomMargin=20*mm,
)

styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "Title", fontName=FONT_NAME, fontSize=16, spaceAfter=6,
    textColor=colors.HexColor("#1a1a2e"), leading=20, alignment=1
)
h1_style = ParagraphStyle(
    "H1", fontName=FONT_NAME, fontSize=13, spaceAfter=4, spaceBefore=12,
    textColor=colors.HexColor("#16213e"), leading=16, borderPad=2
)
h2_style = ParagraphStyle(
    "H2", fontName=FONT_NAME, fontSize=11, spaceAfter=3, spaceBefore=8,
    textColor=colors.HexColor("#0f3460"), leading=14
)
body_style = ParagraphStyle(
    "Body", fontName=FONT_NAME, fontSize=9, spaceAfter=2, leading=13,
    textColor=colors.HexColor("#2d2d2d")
)
code_style = ParagraphStyle(
    "Code", fontName="Courier", fontSize=8, spaceAfter=2, leading=11,
    textColor=colors.HexColor("#1a1a1a"),
    backColor=colors.HexColor("#f5f5f5"), leftIndent=10
)

story = []

with open(REPORT, encoding="utf-8") as f:
    lines = f.readlines()

for line in lines:
    line = line.rstrip("\n")

    if line.startswith("Josh Talks") or line.startswith("Task Submission"):
        story.append(Paragraph(line, title_style))
    elif line.startswith("Submitted by") or line.startswith("Email") or line.startswith("GitHub"):
        story.append(Paragraph(line, body_style))
    elif re.match(r"^={4,}", line):
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#16213e")))
        story.append(Spacer(1, 2*mm))
    elif re.match(r"^-{4,}", line):
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc")))
    elif re.match(r"^QUESTION \d", line) or re.match(r"^Q\d[a-g]? —", line):
        story.append(Spacer(1, 3*mm))
        story.append(Paragraph(line, h1_style))
    elif re.match(r"^[A-Z][A-Z ]+$", line) and len(line) > 4:
        story.append(Paragraph(line, h2_style))
    elif line.startswith("  ") or line.startswith("\t"):
        # indented = code/table style
        safe = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        story.append(Paragraph(safe, code_style))
    elif line.strip() == "":
        story.append(Spacer(1, 2*mm))
    else:
        safe = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        story.append(Paragraph(safe, body_style))

doc.build(story)
size_kb = os.path.getsize(OUTPUT) / 1024
print(f"\nPDF created: {OUTPUT}  ({size_kb:.1f} KB)")
