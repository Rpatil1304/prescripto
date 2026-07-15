"""
translator.py
--------------
Translates structured prescription fields into Hindi and Marathi using
AI4Bharat's IndicTrans2 (distilled 200M variant — chosen specifically
because it's CPU-friendly; the full 1B model realistically needs a GPU).

IMPORTANT: translates only the structured TEXT FIELDS (drug names,
diagnosis, symptoms, frequency/duration phrases), not a rendered image —
per the project plan, the image is regenerated per-language from
translated structured data, not translated as a picture.

Usage:
    from translator import PrescriptionTranslator

    translator = PrescriptionTranslator()
    hindi_data = translator.translate_prescription_data(structured_data, "hin_Deva")
    marathi_data = translator.translate_prescription_data(structured_data, "mar_Deva")
"""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

# Language codes IndicTrans2 expects (FLORES-200 style codes)
LANG_CODES = {
    "english": "eng_Latn",
    "hindi": "hin_Deva",
    "marathi": "mar_Deva",
}

MODEL_NAME = "ai4bharat/indictrans2-en-indic-dist-200M"


class PrescriptionTranslator:
    def __init__(self, model_name=MODEL_NAME, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading IndicTrans2 model on {self.device} (first run downloads ~1GB, may take a few minutes)...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, trust_remote_code=True
        ).to(self.device)
        self.model.eval()
        self.ip = IndicProcessor(inference=True)
        print("Model loaded.")

    def translate_texts(self, texts, target_lang_code, source_lang_code="eng_Latn"):
        """
        texts: list of strings to translate
        target_lang_code: e.g. "hin_Deva" or "mar_Deva"
        Returns: list of translated strings, same order/length as input.
        Empty strings pass through untranslated (no point calling the model
        on nothing, and avoids edge-case errors on empty batches).
        """
        # Separate empty strings out — translate only non-empty ones, then
        # stitch results back into their original positions. This matters
        # because a medicine might have an empty "duration" field (per our
        # generalized, no-fixed-schema design), and empty strings shouldn't
        # be sent to the model.
        non_empty_indices = [i for i, t in enumerate(texts) if t and t.strip()]
        non_empty_texts = [texts[i] for i in non_empty_indices]

        results = list(texts)  # default: pass through as-is (covers empty strings)

        if not non_empty_texts:
            return results

        batch = self.ip.preprocess_batch(
            non_empty_texts, src_lang=source_lang_code, tgt_lang=target_lang_code
        )
        inputs = self.tokenizer(
            batch, padding="longest", truncation=True, max_length=256, return_tensors="pt"
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_length=256, num_beams=5)

        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translated = self.ip.postprocess_batch(decoded, lang=target_lang_code)

        for idx, translation in zip(non_empty_indices, translated):
            results[idx] = translation

        return results

    def translate_prescription_data(self, data: dict, target_lang_code: str) -> dict:
        """
        Takes the structured prescription dict (same format the NER
        connector produces / prescription_generator consumes) and returns
        a new dict with the translatable text fields translated, leaving
        non-linguistic fields (dates, phone numbers) untouched.

        Only translates: diagnosis, symptoms, and each medicine's
        drug/dosage/frequency/duration fields. Clinic/doctor name and
        patient name are left in their original form deliberately —
        proper nouns generally shouldn't be machine-translated.
        """
        translated = dict(data)  # shallow copy; we'll overwrite specific keys

        # Diagnosis + symptoms (lists of strings)
        diagnosis = data.get("diagnosis", []) or []
        symptoms = data.get("symptoms", []) or []
        translated["diagnosis"] = self.translate_texts(diagnosis, target_lang_code)
        translated["symptoms"] = self.translate_texts(symptoms, target_lang_code)

        # Medicines (list of dicts) — translate each field independently
        medicines = data.get("medicines", []) or []
        if medicines:
            drugs = [m.get("drug", "") for m in medicines]
            dosages = [m.get("dosage", "") for m in medicines]
            frequencies = [m.get("frequency", "") for m in medicines]
            durations = [m.get("duration", "") for m in medicines]

            t_drugs = self.translate_texts(drugs, target_lang_code)
            t_dosages = self.translate_texts(dosages, target_lang_code)
            t_frequencies = self.translate_texts(frequencies, target_lang_code)
            t_durations = self.translate_texts(durations, target_lang_code)

            translated["medicines"] = [
                {
                    "drug": t_drugs[i],
                    "dosage": t_dosages[i],
                    "frequency": t_frequencies[i],
                    "duration": t_durations[i],
                }
                for i in range(len(medicines))
            ]

        return translated


if __name__ == "__main__":
    translator = PrescriptionTranslator()

    sample_data = {
        "clinic_name": "City Health Clinic",
        "doctor_name": "Dr. A. Sharma, MBBS",
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

    print("\n--- Original (English) ---")
    print(sample_data)

    hindi_data = translator.translate_prescription_data(sample_data, LANG_CODES["hindi"])
    print("\n--- Hindi ---")
    print(hindi_data)

    marathi_data = translator.translate_prescription_data(sample_data, LANG_CODES["marathi"])
    print("\n--- Marathi ---")
    print(marathi_data)
