"""Doctor input form component."""

from __future__ import annotations

from typing import Dict, List

import streamlit as st


def default_structured_data() -> Dict:
	return {
		"clinic_name": "City Health Clinic",
		"doctor_name": "Dr. A. Sharma, MBBS",
		"clinic_address": "123 MG Road, Pune, Maharashtra",
		"patient_name": "Ramesh Kumar",
		"date": "11 July 2026",
		"age": "45",
		"diagnosis": ["Viral fever"],
		"symptoms": ["Fever", "Body ache"],
		"medicines": [
			{
				"drug": "Paracetamol",
				"dosage": "500 mg",
				"frequency": "twice daily",
				"duration": "5 days",
			},
			{
				"drug": "ORS",
				"dosage": "1 sachet",
				"frequency": "as needed",
				"duration": "3 days",
			},
		],
	}


def _split_values(raw: str) -> List[str]:
	values: List[str] = []
	for item in raw.replace("\n", ",").split(","):
		item = item.strip()
		if item:
			values.append(item)
	return values


def render_manual_prescription_form(key_prefix: str = "manual") -> Dict:
	defaults = default_structured_data()

	st.subheader("Doctor and Patient Details")
	left, right = st.columns(2)
	with left:
		clinic_name = st.text_input(
			"Clinic Name",
			value=defaults["clinic_name"],
			key=f"{key_prefix}_clinic_name",
		)
		doctor_name = st.text_input(
			"Doctor Name",
			value=defaults["doctor_name"],
			key=f"{key_prefix}_doctor_name",
		)
		clinic_address = st.text_area(
			"Clinic Address",
			value=defaults["clinic_address"],
			key=f"{key_prefix}_clinic_address",
		)

	with right:
		patient_name = st.text_input(
			"Patient Name",
			value=defaults["patient_name"],
			key=f"{key_prefix}_patient_name",
		)
		date = st.text_input("Date", value=defaults["date"], key=f"{key_prefix}_date")
		age = st.text_input("Age", value=defaults["age"], key=f"{key_prefix}_age")

	st.subheader("Diagnosis and Symptoms")
	diagnosis_raw = st.text_area(
		"Diagnosis (comma or new line separated)",
		value=", ".join(defaults["diagnosis"]),
		key=f"{key_prefix}_diagnosis",
	)
	symptoms_raw = st.text_area(
		"Symptoms (comma or new line separated)",
		value=", ".join(defaults["symptoms"]),
		key=f"{key_prefix}_symptoms",
	)

	st.subheader("Medicines")
	medicine_count = st.number_input(
		"Number of medicines",
		min_value=1,
		max_value=12,
		value=len(defaults["medicines"]),
		step=1,
		key=f"{key_prefix}_medicine_count",
	)

	medicines: List[Dict[str, str]] = []
	for idx in range(int(medicine_count)):
		fallback = defaults["medicines"][idx] if idx < len(defaults["medicines"]) else {
			"drug": "",
			"dosage": "",
			"frequency": "",
			"duration": "",
		}
		with st.expander(f"Medicine {idx + 1}", expanded=True):
			drug = st.text_input(
				"Drug",
				value=fallback["drug"],
				key=f"{key_prefix}_med_{idx}_drug",
			)
			dosage = st.text_input(
				"Dosage",
				value=fallback["dosage"],
				key=f"{key_prefix}_med_{idx}_dosage",
			)
			frequency = st.text_input(
				"Frequency",
				value=fallback["frequency"],
				key=f"{key_prefix}_med_{idx}_frequency",
			)
			duration = st.text_input(
				"Duration",
				value=fallback["duration"],
				key=f"{key_prefix}_med_{idx}_duration",
			)

		if any([drug.strip(), dosage.strip(), frequency.strip(), duration.strip()]):
			medicines.append(
				{
					"drug": drug.strip(),
					"dosage": dosage.strip(),
					"frequency": frequency.strip(),
					"duration": duration.strip(),
				}
			)

	if not medicines:
		medicines.append({"drug": "", "dosage": "", "frequency": "", "duration": ""})

	return {
		"clinic_name": clinic_name.strip(),
		"doctor_name": doctor_name.strip(),
		"clinic_address": clinic_address.strip(),
		"patient_name": patient_name.strip(),
		"date": date.strip(),
		"age": age.strip(),
		"diagnosis": _split_values(diagnosis_raw),
		"symptoms": _split_values(symptoms_raw),
		"medicines": medicines,
	}
