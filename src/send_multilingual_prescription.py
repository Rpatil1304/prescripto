"""
send_multilingual_prescription.py
-----------------------------------
Ties together Phases 3, 4, and 5: takes a structured prescription (English),
generates English + Hindi + Marathi prescription images, and sends all
three to the patient's WhatsApp number.

This is the piece that actually answers "the prescription should be sent
in Hindi and Marathi too" — previously only the English image was being
sent manually as a one-off test.

Usage:
    from send_multilingual_prescription import send_full_prescription

    send_full_prescription(structured_data, patient_phone="+91XXXXXXXXXX")
"""

import os
from translation.prescripto_translator import PrescriptionTranslator, LANG_CODES
from prescription_generator.prescription_generator import generate_prescription_image
from whatsapp.whatsapp_sender import WhatsAppSender


def send_full_prescription(english_data: dict, patient_phone: str,
                            output_dir="generated_prescriptions"):
    """
    english_data: structured prescription dict (English) — same format
        produced by ner_inference.group_into_prescription_data().
    patient_phone: E.164 format, e.g. "+919876543210"
    output_dir: local folder to save the 3 generated images before upload.

    Returns a dict summarizing what happened, so a failure in one language
    doesn't hide whether the others succeeded — same defensive pattern
    used throughout the rest of this project.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Loading translation model (first run downloads ~1GB, cached after)...")
    translator = PrescriptionTranslator()

    print("Translating to Hindi and Marathi...")
    hindi_data = translator.translate_prescription_data(english_data, LANG_CODES["hindi"])
    marathi_data = translator.translate_prescription_data(english_data, LANG_CODES["marathi"])

    print("Generating prescription images (English, Hindi, Marathi)...")
    image_paths = []
    language_labels = ["English", "Hindi", "Marathi"]

    for data, label in zip([english_data, hindi_data, marathi_data], language_labels):
        img = generate_prescription_image(data)
        path = os.path.join(output_dir, f"prescription_{label.lower()}.png")
        img.save(path)
        image_paths.append(path)
        print(f"  Saved {label} -> {path}")

    print(f"Sending {len(image_paths)} images to {patient_phone} via WhatsApp...")
    sender = WhatsAppSender()
    send_results = sender.send_prescription_images(
        patient_phone, image_paths, language_labels
    )

    for r in send_results:
        status = "SENT" if not r["error"] else "FAILED"
        detail = r["sid"] if not r["error"] else r["error"]
        print(f"  {status}  {r['file_path']}: {detail}")

    return {
        "english_data": english_data,
        "hindi_data": hindi_data,
        "marathi_data": marathi_data,
        "image_paths": image_paths,
        "send_results": send_results,
    }


if __name__ == "__main__":
    sample_data = {
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

    PATIENT_PHONE = "+919518398826"  # replace with your real number

    result = send_full_prescription(sample_data, PATIENT_PHONE)
    print("\nDone.")