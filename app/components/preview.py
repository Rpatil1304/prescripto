"""Prescription preview component."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import streamlit as st


def render_structured_preview(data: Dict) -> None:
	st.subheader("Structured Prescription Data")

	medicines = data.get("medicines", []) or []
	diagnosis = data.get("diagnosis", []) or []
	symptoms = data.get("symptoms", []) or []

	left, right = st.columns(2)
	with left:
		st.markdown(f"Clinic: {data.get('clinic_name', '')}")
		st.markdown(f"Doctor: {data.get('doctor_name', '')}")
		st.markdown(f"Patient: {data.get('patient_name', '')}")
	with right:
		st.markdown(f"Date: {data.get('date', '')}")
		st.markdown(f"Age: {data.get('age', '')}")
		st.markdown(f"Medicines: {len(medicines)}")

	if diagnosis:
		st.markdown("Diagnosis")
		for item in diagnosis:
			st.write(f"- {item}")

	if symptoms:
		st.markdown("Symptoms")
		for item in symptoms:
			st.write(f"- {item}")

	if medicines:
		st.markdown("Medicines")
		st.dataframe(medicines, use_container_width=True)

	with st.expander("View Raw JSON"):
		st.json(data)


def render_entities_preview(entities: List[tuple]) -> None:
	if not entities:
		return
	st.subheader("Extracted NER Entities")
	st.dataframe(
		[{"entity_type": label, "entity_text": text} for label, text in entities],
		use_container_width=True,
	)


def render_run_summary(result: Dict) -> None:
	st.subheader("Run Summary")
	image_paths = result.get("image_paths", []) or []
	send_results = result.get("send_results", []) or []

	st.write(f"Generated images: {len(image_paths)}")

	if send_results:
		rows = []
		for item in send_results:
			rows.append(
				{
					"file": item.get("file_path", ""),
					"status": "SENT" if not item.get("error") else "FAILED",
					"sid": item.get("sid", ""),
					"error": item.get("error", ""),
				}
			)
		st.dataframe(rows, use_container_width=True)


def render_generated_images(image_paths: List[str]) -> None:
	if not image_paths:
		return

	st.subheader("Generated Prescription Images")
	for path in image_paths:
		if Path(path).exists():
			st.image(path, caption=Path(path).name, use_container_width=True)
		else:
			st.warning(f"Image not found: {path}")
