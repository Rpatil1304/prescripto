"""Single entrypoint for the full Prescripto pipeline.

This script supports two modes:
- demo mode: uses a built-in sample prescription
- free-text mode: extracts structured data from raw prescription text via the
  trained NER model, then translates, renders, and sends the prescription in
  English, Hindi, and Marathi
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
NER_DIR = SRC_DIR / "ner"

for path in (SRC_DIR, NER_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


DEFAULT_PHONE_NUMBER = "+919518398826"
DEFAULT_SAMPLE_DATA = {
    "clinic_name": "City Health Clinic",
    "doctor_name": "Dr. A. Sharma, MBBS",
    "clinic_address": "123 MG Road, Pune, Maharashtra",
    "patient_name": "Ramesh Kumar",
    "date": "11 July 2026",
    "age": "45",
    "diagnosis": ["Viral fever"],
    "symptoms": ["Fever", "Body ache"],
    "medicines": [
        {"drug": "Paracetamol", "dosage": "500 mg", "frequency": "twice daily", "duration": "5 days"},
        {"drug": "ORS", "dosage": "1 sachet", "frequency": "as needed", "duration": "3 days"},
    ],
}
DEFAULT_RAW_TEXT = (
    "Patient has viral fever. Give Paracetamol 500 mg twice daily for 5 days. "
    "ORS as needed for dehydration."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full Prescripto pipeline")
    parser.add_argument("--phone", default=DEFAULT_PHONE_NUMBER, help="Patient WhatsApp number in E.164 format")
    parser.add_argument("--text", default="", help="Raw prescription text to extract entities from")
    parser.add_argument("--patient-name", default="Ramesh Kumar")
    parser.add_argument("--date", default="11 July 2026")
    parser.add_argument("--age", default="45")
    parser.add_argument("--clinic-name", default="City Health Clinic")
    parser.add_argument("--doctor-name", default="Dr. A. Sharma, MBBS")
    parser.add_argument("--clinic-address", default="123 MG Road, Pune, Maharashtra")
    parser.add_argument("--model-dir", default="models/ner_model", help="Directory containing the trained NER model")
    parser.add_argument("--output-dir", default="generated_prescriptions", help="Folder for generated prescription images")
    parser.add_argument("--generate-only", action="store_true", help="Generate prescription images but do not send to WhatsApp")
    return parser.parse_args()


def build_structured_data_from_text(raw_text: str, args: argparse.Namespace) -> dict:
    from ner_inference import NERModel, group_into_prescription_data

    print("Loading NER model...")
    ner = NERModel(args.model_dir)

    print("Extracting entities from raw text...")
    entities = ner.extract_entities(raw_text)
    print(f"  Extracted {len(entities)} entity span(s)")

    return group_into_prescription_data(
        entities,
        patient_name=args.patient_name,
        date=args.date,
        age=args.age,
        clinic_name=args.clinic_name,
        doctor_name=args.doctor_name,
        clinic_address=args.clinic_address,
    )


def load_pipeline_function():
    from send_multilingual_prescription import send_full_prescription

    return send_full_prescription


def main() -> dict:
    args = parse_args()
    send_full_prescription = load_pipeline_function()

    if args.text.strip():
        structured_data = build_structured_data_from_text(args.text, args)
    else:
        print("No raw text supplied; using the built-in demo prescription.")
        structured_data = dict(DEFAULT_SAMPLE_DATA)

    if args.generate_only:
        from translation.prescripto_translator import PrescriptionTranslator, LANG_CODES
        from prescription_generator.prescription_generator import generate_prescription_image

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("Loading translation model (first run downloads ~1GB, cached after)...")
        translator = PrescriptionTranslator()

        print("Translating to Hindi and Marathi...")
        hindi_data = translator.translate_prescription_data(structured_data, LANG_CODES["hindi"])
        marathi_data = translator.translate_prescription_data(structured_data, LANG_CODES["marathi"])

        print("Generating prescription images (English, Hindi, Marathi)...")
        image_paths = []
        for data, label in zip([structured_data, hindi_data, marathi_data], ["English", "Hindi", "Marathi"]):
            image = generate_prescription_image(data)
            path = output_dir / f"prescription_{label.lower()}.png"
            image.save(path)
            image_paths.append(str(path))
            print(f"  Saved {label} -> {path}")

        return {
            "structured_data": structured_data,
            "image_paths": image_paths,
            "send_results": [],
        }

    result = send_full_prescription(structured_data, args.phone, args.output_dir)
    return result


if __name__ == "__main__":
    main()