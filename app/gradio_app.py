"""Gradio entry point for Prescripto — simple single-form doctor UI.

Doctor fills in patient details + prescription text, clicks one button,
and the full backend pipeline (NER extraction -> translation -> image
generation -> WhatsApp delivery) runs automatically. No intermediate
screens, no debug output shown — just a final status message.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import gradio as gr
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
NER_DIR = SRC_DIR / "ner"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "generated_prescriptions" / "records"
MODEL_DIR = str(ROOT_DIR / "models" / "ner_model")


def _ensure_import_paths() -> None:
    for path in (ROOT_DIR, SRC_DIR, NER_DIR):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


load_dotenv(ROOT_DIR / ".env")


def _validate_whatsapp_env() -> List[str]:
    required = [
        "TWILIO_ACCOUNT_SID",
        "TWILIO_AUTH_TOKEN",
        "CLOUDINARY_CLOUD_NAME",
        "CLOUDINARY_API_KEY",
        "CLOUDINARY_API_SECRET",
    ]
    return [key for key in required if not os.environ.get(key)]


def _build_run_output_dir(patient_name: str) -> Path:
    import re

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", patient_name.strip()).strip("_") or "patient"
    run_dir = DEFAULT_OUTPUT_DIR / f"{timestamp}_{safe_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


@lru_cache(maxsize=1)
def _load_ner_model(model_dir: str):
    from ner_inference import NERModel

    return NERModel(model_dir)


def generate_and_send(
    patient_name: str,
    age: str,
    symptoms_diagnosis: str,
    prescription: str,
    phone: str,
    doctor_name: str,
    clinic_name: str,
    clinic_address: str,
) -> str:
    """
    Single entry point triggered by the one form button. Runs the entire
    pipeline and returns a plain-language status message — no intermediate
    data (entities, structured JSON, image previews) is shown to the
    doctor, per the requirement to keep the UI simple.
    """
    # --- Validate required fields ---
    patient_name = (patient_name or "").strip()
    phone = (phone or "").strip()
    prescription = (prescription or "").strip()

    if not patient_name:
        return "⚠️ Please enter the patient's name."
    if not phone:
        return "⚠️ Please enter the patient's WhatsApp number (with country code, e.g. +91...)."
    if not prescription:
        return "⚠️ Please enter the prescription text."

    missing_keys = _validate_whatsapp_env()
    if missing_keys:
        return (
            "⚠️ Server configuration error — missing: " + ", ".join(missing_keys) +
            ". Please contact your administrator."
        )

    try:
        from ner_inference import group_into_prescription_data
        from send_multilingual_prescription import send_full_prescription

        # --- Run NER on symptoms/diagnosis + prescription text together ---
        ner = _load_ner_model(MODEL_DIR)
        combined_text = f"{symptoms_diagnosis.strip()}\n{prescription}".strip()
        entities = ner.extract_entities(combined_text)

        structured_data = group_into_prescription_data(
            entities,
            patient_name=patient_name,
            date=datetime.now().strftime("%d %B %Y"),
            age=(age or "").strip(),
            clinic_name=(clinic_name or "").strip(),
            doctor_name=(doctor_name or "").strip(),
            clinic_address=(clinic_address or "").strip(),
        )

        # --- Guard: if NER found no medicines at all, don't silently send an empty prescription ---
        if not structured_data.get("medicines"):
            return (
                "⚠️ No medicines could be detected in the prescription text. "
                "Please check the wording and try again."
            )

        run_dir = _build_run_output_dir(patient_name)

        # --- Translate, generate 3 images, upload, and send via WhatsApp ---
        result = send_full_prescription(structured_data, phone, str(run_dir))
        send_results = result.get("send_results", [])

        # --- Build a simple, doctor-friendly status summary ---
        sent = [r for r in send_results if not r.get("error")]
        failed = [r for r in send_results if r.get("error")]

        if len(sent) == len(send_results) and send_results:
            return f"✅ Prescription sent successfully to {phone} (English, Hindi, Marathi)."
        elif sent and failed:
            return (
                f"⚠️ Prescription partially sent to {phone}. "
                f"{len(sent)} of {len(send_results)} languages delivered successfully."
            )
        else:
            return f"❌ Failed to send prescription to {phone}. Please try again."

    except Exception as e:
        return f"❌ An error occurred: {e}"


def launch() -> None:
    _ensure_import_paths()

    with gr.Blocks(
        title="Prescripto",
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css="""
        .app-shell { max-width: 640px; margin: 0 auto; }
        .hero h1 { font-size: 2rem; margin-bottom: 0.25rem; }
        .hero p { color: #475569; margin-top: 0; }
        """,
    ) as demo:
        with gr.Column(elem_classes=["app-shell"]):
            gr.Markdown("# Prescripto", elem_classes=["hero"])
            gr.Markdown("Enter patient and prescription details, then click Send.")

            patient_name = gr.Textbox(label="Patient Name", placeholder="e.g. Ramesh Kumar")
            age = gr.Textbox(label="Age (optional)", placeholder="e.g. 45")
            symptoms_diagnosis = gr.Textbox(
                label="Symptoms / Diagnosis",
                placeholder="e.g. Patient has viral fever and body ache.",
                lines=2,
            )
            prescription = gr.Textbox(
                label="Prescription",
                placeholder="e.g. Paracetamol 500 mg twice daily for 5 days.",
                lines=4,
            )
            phone = gr.Textbox(label="Patient WhatsApp Number", placeholder="e.g. +919876543210")

            with gr.Accordion("Doctor / Clinic Details (optional)", open=False):
                doctor_name = gr.Textbox(label="Doctor Name", placeholder="e.g. Dr. A. Sharma, MBBS")
                clinic_name = gr.Textbox(label="Clinic Name", placeholder="e.g. City Health Clinic")
                clinic_address = gr.Textbox(label="Clinic Address", placeholder="e.g. 123 MG Road, Pune")

            submit_btn = gr.Button("Generate & Send Prescription", variant="primary")
            status = gr.Textbox(label="Status", interactive=False)

            submit_btn.click(
                generate_and_send,
                inputs=[
                    patient_name,
                    age,
                    symptoms_diagnosis,
                    prescription,
                    phone,
                    doctor_name,
                    clinic_name,
                    clinic_address,
                ],
                outputs=[status],
            )

    demo.launch()


if __name__ == "__main__":
    launch()