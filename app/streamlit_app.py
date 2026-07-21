"""Streamlit entry point for Prescripto."""

from __future__ import annotations

import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
NER_DIR = SRC_DIR / "ner"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "generated_prescriptions" / "records"


load_dotenv(ROOT_DIR / ".env")


def _ensure_import_paths() -> None:
    for path in (ROOT_DIR, SRC_DIR, NER_DIR):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


@st.cache_resource(show_spinner=False)
def _load_ner_model(model_dir: str):
    from ner_inference import NERModel

    return NERModel(model_dir)


@st.cache_resource(show_spinner=False)
def _load_translator():
    from translation.prescripto_translator import PrescriptionTranslator

    return PrescriptionTranslator()


def _build_structured_data_from_text(raw_text: str, meta: Dict, model_dir: str) -> Tuple[Dict, List[tuple]]:
    from ner_inference import group_into_prescription_data

    ner = _load_ner_model(model_dir)
    entities = ner.extract_entities(raw_text)
    structured_data = group_into_prescription_data(
        entities,
        patient_name=meta["patient_name"],
        date=meta["date"],
        age=meta["age"],
        clinic_name=meta["clinic_name"],
        doctor_name=meta["doctor_name"],
        clinic_address=meta["clinic_address"],
    )
    return structured_data, entities


def _validate_whatsapp_env() -> List[str]:
    required = [
        "TWILIO_ACCOUNT_SID",
        "TWILIO_AUTH_TOKEN",
        "CLOUDINARY_CLOUD_NAME",
        "CLOUDINARY_API_KEY",
        "CLOUDINARY_API_SECRET",
    ]
    return [key for key in required if not os.environ.get(key)]


def _sanitize_filename_part(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip())
    return cleaned.strip("_") or "patient"


def _build_run_output_dir(patient_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = _sanitize_filename_part(patient_name)
    run_dir = DEFAULT_OUTPUT_DIR / f"{timestamp}_{safe_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _render_record_downloads(record: Dict) -> None:
    image_paths = record.get("image_paths", [])
    labels = ["English", "Hindi", "Marathi"]
    cols = st.columns(3)

    for idx, path_str in enumerate(image_paths[:3]):
        path = Path(path_str)
        if not path.exists():
            continue
        with cols[idx]:
            with open(path, "rb") as f:
                st.download_button(
                    label=f"Download {labels[idx]}",
                    data=f.read(),
                    file_name=path.name,
                    mime="image/png",
                    key=f"download_{record['id']}_{idx}",
                    use_container_width=True,
                )


def _render_records() -> None:
    records = st.session_state.get("records", [])
    if not records:
        return

    st.subheader("Doctor Records")
    st.caption("Stored copies from previous sends in this session.")

    for record in records:
        summary = (
            f"{record['created_at']} | {record['patient_name']} | {record['phone']}"
        )
        with st.expander(summary, expanded=False):
            send_results = record.get("send_results", [])
            for idx, label in enumerate(["English", "Hindi", "Marathi"]):
                status = "Sent"
                if idx < len(send_results) and send_results[idx].get("error"):
                    status = "Failed"
                st.write(f"{label}: {status}")
            _render_record_downloads(record)


def main() -> None:
    _ensure_import_paths()

    st.set_page_config(page_title="Prescripto", page_icon="Rx", layout="centered")
    st.title("Generate and Send Prescription")
    st.caption("Fill the form, click once, and wait for completion.")

    st.sidebar.header("Technical Settings")
    model_dir = st.sidebar.text_input("NER Model Directory", value="models/ner_model")

    if "records" not in st.session_state:
        st.session_state["records"] = []

    with st.form("doctor_prescription_form", clear_on_submit=True):
        st.subheader("Doctor and Clinic")
        doctor_name = st.text_input("Doctor Name")
        clinic_name = st.text_input("Clinic Name")
        clinic_address = st.text_area("Clinic Address", height=80)

        st.subheader("Patient")
        patient_name = st.text_input("Patient Name")
        age = st.text_input("Age")
        phone = st.text_input("WhatsApp Number", placeholder="+91XXXXXXXXXX")

        st.subheader("Clinical Notes")
        symptoms_diagnosis = st.text_area(
            "Symptoms / Diagnosis",
            help="Free text from consultation notes.",
            height=100,
        )
        prescription_text = st.text_area(
            "Prescription",
            help="Medicine instructions in free text.",
            height=120,
        )

        submitted = st.form_submit_button(
            "Generate & Send Prescription",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if not patient_name.strip() or not phone.strip() or not prescription_text.strip():
            st.error("Patient Name, WhatsApp Number, and Prescription are required.")
            _render_records()
            return

        missing_keys = _validate_whatsapp_env()
        if missing_keys:
            st.error(
                "Missing required environment variables for WhatsApp sending: "
                + ", ".join(missing_keys)
            )
            _render_records()
            return

        combined_text = f"{symptoms_diagnosis.strip()}\n{prescription_text.strip()}".strip()
        meta = {
            "patient_name": patient_name.strip(),
            "date": datetime.now().strftime("%d %B %Y"),
            "age": age.strip(),
            "clinic_name": clinic_name.strip(),
            "doctor_name": doctor_name.strip(),
            "clinic_address": clinic_address.strip(),
        }

        from send_multilingual_prescription import send_full_prescription

        with st.spinner("Processing NER, translation, image generation, upload, and WhatsApp send..."):
            structured_data, _ = _build_structured_data_from_text(
                raw_text=combined_text,
                meta=meta,
                model_dir=model_dir,
            )
            run_output_dir = _build_run_output_dir(patient_name=patient_name)
            result = send_full_prescription(
                english_data=structured_data,
                patient_phone=phone.strip(),
                output_dir=str(run_output_dir),
            )

        send_results = result.get("send_results", [])
        all_sent = bool(send_results) and all(not item.get("error") for item in send_results)

        if all_sent:
            st.success(f"Prescription sent to {phone.strip()}")
        else:
            st.warning(f"Prescription process finished for {phone.strip()} with partial failures")

        st.write("Delivery Status")
        for idx, label in enumerate(["English", "Hindi", "Marathi"]):
            status = "Sent"
            if idx < len(send_results) and send_results[idx].get("error"):
                status = "Failed"
            st.write(f"{label}: {status}")

        record = {
            "id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "patient_name": patient_name.strip(),
            "phone": phone.strip(),
            "image_paths": result.get("image_paths", []),
            "send_results": send_results,
        }
        st.session_state["records"].insert(0, record)

    _render_records()


if __name__ == "__main__":
    main()
