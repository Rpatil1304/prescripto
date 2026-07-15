"""
prescription_generator.py
--------------------------
Renders a structured prescription (as produced by the NER model's output)
into a clean, professional prescription image using Pillow.

Design principle (per project requirement): the layout is fully dynamic.
It does NOT assume a fixed number of medicines, a fixed amount of
diagnosis/symptom text, or any other fixed-size input. A prescription with
1 medicine and a prescription with 10 medicines both render correctly,
with the canvas height and internal spacing computed from the actual
content — never hardcoded or template-truncated.

Usage:
    from prescription_generator import generate_prescription_image

    data = {
        "clinic_name": "City Health Clinic",
        "doctor_name": "Dr. A. Sharma, MBBS",
        "clinic_address": "123 MG Road, Pune, Maharashtra",
        "patient_name": "Ramesh Kumar",
        "date": "10 July 2026",
        "age": "45",
        "diagnosis": ["Viral fever", "Mild dehydration"],
        "symptoms": ["Fever", "Body ache"],
        "medicines": [
            {"drug": "Paracetamol", "dosage": "500 mg", "frequency": "twice daily", "duration": "5 days"},
            {"drug": "ORS", "dosage": "1 sachet", "frequency": "as needed", "duration": "3 days"},
        ],
    }
    img = generate_prescription_image(data)
    img.save("prescription.png")
"""

from PIL import Image, ImageDraw, ImageFont
import os
import textwrap


# ---------------------------------------------------------------------------
# Font handling. Falls back gracefully if the preferred fonts aren't
# installed on the machine running this script (e.g. on Windows without
# DejaVu fonts) — uses Pillow's built-in default font as a last resort so
# the script never crashes purely due to a missing font file.
# ---------------------------------------------------------------------------

FONT_CANDIDATES = {
    "regular": [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ],
    "bold": [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ],
    "italic": [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
        "C:/Windows/Fonts/ariali.ttf",
        "/System/Library/Fonts/Supplemental/Arial Italic.ttf",
    ],
}

# Devanagari script (used by Hindi and Marathi) needs a different font —
# Arial/DejaVu don't include Devanagari glyphs and would render blank boxes
# or "tofu" characters instead of actual text. Noto Sans Devanagari is a
# free, open-license Google font with full Devanagari support. Place the
# .ttf files in the assets/fonts/ folder alongside this script (see
# assets/fonts/README.md for the download link) — this candidate list
# checks a few common install locations too, in case the font happens to
# already be installed system-wide.
FONT_CANDIDATES_DEVANAGARI = {
    "regular": [
        os.path.join(os.path.dirname(__file__), "assets", "fonts", "NotoSansDevanagari-Regular.ttf"),
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
        "C:/Windows/Fonts/NotoSansDevanagari-Regular.ttf",
    ],
    "bold": [
        os.path.join(os.path.dirname(__file__), "assets", "fonts", "NotoSansDevanagari-Bold.ttf"),
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf",
        "C:/Windows/Fonts/NotoSansDevanagari-Bold.ttf",
    ],
    # Noto Sans Devanagari ships without a separate italic/oblique variant
    # in the basic static set, so italic-style text (e.g. the clinic
    # address line) falls back to regular weight for Devanagari content.
    "italic": [
        os.path.join(os.path.dirname(__file__), "assets", "fonts", "NotoSansDevanagari-Regular.ttf"),
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
        "C:/Windows/Fonts/NotoSansDevanagari-Regular.ttf",
    ],
}


def _contains_devanagari(text: str) -> bool:
    """Detects whether a string contains any Devanagari script characters
    (Unicode block U+0900-U+097F), used to auto-select the right font."""
    return any("\u0900" <= ch <= "\u097F" for ch in text)


def _load_font(style, size, use_devanagari=False):
    candidates = FONT_CANDIDATES_DEVANAGARI[style] if use_devanagari else FONT_CANDIDATES[style]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    # Last-resort fallback: try the non-Devanagari candidates too, then
    # Pillow's built-in default (better to show *something* than crash).
    if use_devanagari:
        for path in FONT_CANDIDATES[style]:
            try:
                return ImageFont.truetype(path, size)
            except (OSError, IOError):
                continue
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

CANVAS_WIDTH = 900
MARGIN = 50
CONTENT_WIDTH = CANVAS_WIDTH - 2 * MARGIN

COLOR_TEXT = (30, 30, 30)
COLOR_MUTED = (110, 110, 110)
COLOR_ACCENT = (20, 90, 130)
COLOR_LINE = (200, 200, 200)
COLOR_BG = (255, 255, 255)

FONT_SIZES = {
    "clinic_name": 30,
    "doctor_name": 16,
    "address": 13,
    "section_label": 14,
    "body": 15,
    "rx_symbol": 34,
    "footer": 13,
}


def _wrap_text(draw, text, font, max_width):
    """Word-wrap text to fit max_width, returns list of lines."""
    if not text:
        return []
    words = text.split()
    lines, current = [], ""
    for word in words:
        candidate = (current + " " + word).strip()
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _measure_lines_height(lines, font, line_spacing=6):
    if not lines:
        return 0
    # approximate line height from font size; consistent across this script
    ascent, descent = font.getmetrics()
    line_height = ascent + descent + line_spacing
    return line_height * len(lines)


def generate_prescription_image(data: dict) -> Image.Image:
    """
    data keys (all optional except patient_name and medicines, to support
    varied/incomplete real-world input rather than requiring a rigid schema):
        clinic_name, doctor_name, clinic_address, patient_name, date, age,
        diagnosis (list[str]), symptoms (list[str]),
        medicines (list[dict] with keys: drug, dosage, frequency, duration)
    """
    clinic_name = data.get("clinic_name", "")
    doctor_name = data.get("doctor_name", "")
    clinic_address = data.get("clinic_address", "")
    patient_name = data.get("patient_name", "Patient")
    date = data.get("date", "")
    age = data.get("age", "")
    diagnosis = data.get("diagnosis", []) or []
    symptoms = data.get("symptoms", []) or []
    medicines = data.get("medicines", []) or []

    # Detect whether ANY content in this prescription contains Devanagari
    # script (Hindi/Marathi) — if so, use the Devanagari-capable font
    # throughout the whole image for consistency, since a prescription is
    # normally entirely in one language, not mixed per-field.
    all_text_content = " ".join([
        clinic_name, doctor_name, clinic_address, patient_name, date, age,
        " ".join(diagnosis), " ".join(symptoms),
        " ".join(_format_medicine_line(m) for m in medicines),
    ])
    use_devanagari = _contains_devanagari(all_text_content)

    font_clinic = _load_font("bold", FONT_SIZES["clinic_name"], use_devanagari)
    font_doctor = _load_font("regular", FONT_SIZES["doctor_name"], use_devanagari)
    font_address = _load_font("italic", FONT_SIZES["address"], use_devanagari)
    font_label = _load_font("bold", FONT_SIZES["section_label"], use_devanagari)
    font_body = _load_font("regular", FONT_SIZES["body"], use_devanagari)
    font_body_bold = _load_font("bold", FONT_SIZES["body"], use_devanagari)
    font_rx = _load_font("bold", FONT_SIZES["rx_symbol"], use_devanagari)
    font_footer = _load_font("italic", FONT_SIZES["footer"], use_devanagari)

    # --- Pass 1: measure content on a scratch canvas to compute total height ---
    scratch = Image.new("RGB", (CANVAS_WIDTH, 10), COLOR_BG)
    draw = ImageDraw.Draw(scratch)

    y = MARGIN

    # Header block
    y += FONT_SIZES["clinic_name"] + 10
    if doctor_name:
        y += FONT_SIZES["doctor_name"] + 6
    if clinic_address:
        addr_lines = _wrap_text(draw, clinic_address, font_address, CONTENT_WIDTH)
        y += _measure_lines_height(addr_lines, font_address)
    y += 20  # divider spacing

    # Patient info row
    y += FONT_SIZES["body"] + 20

    # Diagnosis / symptoms (only if present — no fixed slot reserved otherwise)
    if diagnosis:
        y += FONT_SIZES["section_label"] + 4
        diag_text = ", ".join(diagnosis)
        diag_lines = _wrap_text(draw, diag_text, font_body, CONTENT_WIDTH)
        y += _measure_lines_height(diag_lines, font_body) + 10
    if symptoms:
        y += FONT_SIZES["section_label"] + 4
        symp_text = ", ".join(symptoms)
        symp_lines = _wrap_text(draw, symp_text, font_body, CONTENT_WIDTH)
        y += _measure_lines_height(symp_lines, font_body) + 10

    # Rx section
    y += FONT_SIZES["rx_symbol"] + 15

    medicine_row_heights = []
    for med in medicines:
        line = _format_medicine_line(med)
        lines = _wrap_text(draw, line, font_body, CONTENT_WIDTH - 40)
        h = _measure_lines_height(lines, font_body) + 14
        medicine_row_heights.append(h)
        y += h

    if not medicines:
        y += FONT_SIZES["body"] + 14  # "No medicines listed" fallback line

    # Footer (signature area)
    y += 60 + FONT_SIZES["footer"] + 20

    total_height = y + MARGIN

    # --- Pass 2: actually draw on a correctly-sized canvas ---
    img = Image.new("RGB", (CANVAS_WIDTH, total_height), COLOR_BG)
    draw = ImageDraw.Draw(img)

    y = MARGIN

    # Header
    draw.text((MARGIN, y), clinic_name or "Prescription", font=font_clinic, fill=COLOR_ACCENT)
    y += FONT_SIZES["clinic_name"] + 10
    if doctor_name:
        draw.text((MARGIN, y), doctor_name, font=font_doctor, fill=COLOR_TEXT)
        y += FONT_SIZES["doctor_name"] + 6
    if clinic_address:
        addr_lines = _wrap_text(draw, clinic_address, font_address, CONTENT_WIDTH)
        for line in addr_lines:
            draw.text((MARGIN, y), line, font=font_address, fill=COLOR_MUTED)
            y += font_address.getmetrics()[0] + font_address.getmetrics()[1] + 6
    y += 8
    draw.line([(MARGIN, y), (CANVAS_WIDTH - MARGIN, y)], fill=COLOR_LINE, width=2)
    y += 20

    # Patient info row (name left, date + age right)
    draw.text((MARGIN, y), f"Patient: {patient_name}", font=font_body_bold, fill=COLOR_TEXT)
    right_text = date
    if age:
        right_text = f"Age: {age}    {date}" if date else f"Age: {age}"
    if right_text:
        bbox = draw.textbbox((0, 0), right_text, font=font_body)
        draw.text((CANVAS_WIDTH - MARGIN - (bbox[2] - bbox[0]), y), right_text, font=font_body, fill=COLOR_MUTED)
    y += FONT_SIZES["body"] + 20

    # Diagnosis
    if diagnosis:
        draw.text((MARGIN, y), "Diagnosis:", font=font_label, fill=COLOR_ACCENT)
        y += FONT_SIZES["section_label"] + 4
        diag_text = ", ".join(diagnosis)
        diag_lines = _wrap_text(draw, diag_text, font_body, CONTENT_WIDTH)
        for line in diag_lines:
            draw.text((MARGIN, y), line, font=font_body, fill=COLOR_TEXT)
            y += font_body.getmetrics()[0] + font_body.getmetrics()[1] + 6
        y += 10

    # Symptoms
    if symptoms:
        draw.text((MARGIN, y), "Symptoms:", font=font_label, fill=COLOR_ACCENT)
        y += FONT_SIZES["section_label"] + 4
        symp_text = ", ".join(symptoms)
        symp_lines = _wrap_text(draw, symp_text, font_body, CONTENT_WIDTH)
        for line in symp_lines:
            draw.text((MARGIN, y), line, font=font_body, fill=COLOR_TEXT)
            y += font_body.getmetrics()[0] + font_body.getmetrics()[1] + 6
        y += 10

    # Rx section
    draw.text((MARGIN, y), "Rx", font=font_rx, fill=COLOR_ACCENT)
    y += FONT_SIZES["rx_symbol"] + 15

    if medicines:
        for i, med in enumerate(medicines, start=1):
            line = _format_medicine_line(med)
            lines = _wrap_text(draw, line, font_body, CONTENT_WIDTH - 40)
            prefix = f"{i}. "
            draw.text((MARGIN, y), prefix, font=font_body_bold, fill=COLOR_TEXT)
            prefix_width = draw.textbbox((0, 0), prefix, font=font_body_bold)[2]
            for j, l in enumerate(lines):
                x_offset = MARGIN + prefix_width if j == 0 else MARGIN + 40
                draw.text((x_offset, y), l, font=font_body, fill=COLOR_TEXT)
                y += font_body.getmetrics()[0] + font_body.getmetrics()[1] + 6
            y += 8
    else:
        draw.text((MARGIN, y), "No medicines listed.", font=font_body, fill=COLOR_MUTED)
        y += FONT_SIZES["body"] + 14

    # Footer / signature
    y += 40
    draw.line([(CANVAS_WIDTH - MARGIN - 220, y), (CANVAS_WIDTH - MARGIN, y)], fill=COLOR_LINE, width=1)
    y += 6
    draw.text((CANVAS_WIDTH - MARGIN - 220, y), "Doctor's Signature", font=font_footer, fill=COLOR_MUTED)

    return img


def _format_medicine_line(med: dict) -> str:
    """Builds a display line from a medicine dict, gracefully handling
    missing fields (a medicine entry might not have a duration, etc.)."""
    parts = [med.get("drug", "").strip()]
    for key in ("dosage", "frequency", "duration"):
        val = med.get(key)
        if val:
            parts.append(str(val).strip())
    return "  |  ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Demo / self-test: generate two prescriptions with very different amounts
# of content, to verify the dynamic layout genuinely holds up rather than
# only working for one "ideal" example.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    minimal_case = {
        "clinic_name": "City Health Clinic",
        "doctor_name": "Dr. A. Sharma, MBBS",
        "clinic_address": "123 MG Road, Pune, Maharashtra",
        "patient_name": "Ramesh Kumar",
        "date": "10 July 2026",
        "age": "45",
        "diagnosis": ["Viral fever"],
        "medicines": [
            {"drug": "Paracetamol", "dosage": "500 mg", "frequency": "twice daily", "duration": "5 days"},
        ],
    }

    dense_case = {
        "clinic_name": "Wellness Family Care Center",
        "doctor_name": "Dr. Priya Deshmukh, MD (Internal Medicine)",
        "clinic_address": "45 FC Road, Near Deccan Gymkhana, Pune, Maharashtra 411004",
        "patient_name": "Sunita Devi Patil",
        "date": "10 July 2026",
        "age": "68",
        "diagnosis": [
            "Type 2 Diabetes Mellitus",
            "Hypertension",
            "Mild Osteoarthritis of both knees",
        ],
        "symptoms": ["Joint pain", "Fatigue", "Occasional dizziness"],
        "medicines": [
            {"drug": "Metformin", "dosage": "500 mg", "frequency": "twice daily", "duration": "30 days"},
            {"drug": "Amlodipine", "dosage": "5 mg", "frequency": "once daily", "duration": "30 days"},
            {"drug": "Atorvastatin", "dosage": "10 mg", "frequency": "once daily at night", "duration": "30 days"},
            {"drug": "Glimepiride", "dosage": "2 mg", "frequency": "once daily before breakfast", "duration": "30 days"},
            {"drug": "Paracetamol", "dosage": "650 mg", "frequency": "as needed for pain, max 3 times daily", "duration": "10 days"},
            {"drug": "Calcium + Vitamin D3", "dosage": "1 tablet", "frequency": "once daily", "duration": "60 days"},
            {"drug": "Omeprazole", "dosage": "20 mg", "frequency": "once daily before breakfast", "duration": "15 days"},
            {"drug": "Physiotherapy referral", "dosage": "", "frequency": "3 sessions per week", "duration": "4 weeks"},
        ],
    }

    empty_medicines_case = {
        "clinic_name": "City Health Clinic",
        "patient_name": "Test Patient",
        "medicines": [],
    }

    img1 = generate_prescription_image(minimal_case)
    img1.save("prescription_minimal.png")
    print(f"Minimal case: {img1.size}")

    img2 = generate_prescription_image(dense_case)
    img2.save("prescription_dense.png")
    print(f"Dense case: {img2.size}")

    img3 = generate_prescription_image(empty_medicines_case)
    img3.save("prescription_empty_medicines.png")
    print(f"Empty-medicines edge case: {img3.size}")
