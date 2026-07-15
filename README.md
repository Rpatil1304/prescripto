# Prescripto

Prescripto is a multilingual prescription workflow that converts structured or free-text medical input into prescription images in English, Hindi, and Marathi, then sends them to a patient over WhatsApp.

## About The Project

The goal of this project is to reduce the gap between handwritten clinic prescriptions and patient-readable digital prescriptions. It takes medical text from the doctor, structures it, translates it into Indian languages, renders clean prescription images, and delivers them through WhatsApp.

## How We Built It

The project was implemented as an end-to-end Python pipeline:

1. Collected and prepared clinical text data for NER training.
2. Trained a BiLSTM-CRF model to extract entities such as drug, dosage, frequency, duration, diagnosis, and symptoms.
3. Converted the extracted entities into structured prescription data.
4. Translated the structured fields into Hindi and Marathi with IndicTrans2.
5. Rendered language-specific prescription images with Pillow.
6. Uploaded the images to Cloudinary and sent them to WhatsApp using Twilio.

## Pipeline

1. Extract entities from raw prescription text with the trained BiLSTM-CRF NER model.
2. Group the extracted entities into structured prescription data.
3. Translate the structured fields into Hindi and Marathi.
4. Render three prescription images with Pillow.
5. Upload the images to Cloudinary.
6. Send the images to WhatsApp through the Twilio WhatsApp API.

## Repository Layout

- `main.py` - single entrypoint for the full pipeline
- `src/send_multilingual_prescription.py` - orchestration helper for translation, rendering, and WhatsApp delivery
- `src/ner/` - NER inference, preprocessing, training, and evaluation code
- `src/translation/` - IndicTrans2 translation helpers
- `src/prescription_generator/` - prescription image renderer
- `src/whatsapp/` - Cloudinary upload and Twilio WhatsApp sender
- `app/streamlit_app.py` - Streamlit entrypoint placeholder
- `notebooks/` - training and preprocessing notebooks
- `models/ner_model/` - trained NER model artifacts
- `outputs/` and `generated_prescriptions/` - generated files and intermediate artifacts

## Setup

Create a virtual environment and install the project dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The full pipeline also depends on the ML/runtime packages used by the source modules, including TensorFlow, PyTorch, Transformers, IndicTransToolkit, Twilio, Cloudinary, and Pillow. Install them if they are not already available in your environment.

Set the required environment variables before sending messages:

- `TWILIO_ACCOUNT_SID`
- `TWILIO_AUTH_TOKEN`
- `CLOUDINARY_CLOUD_NAME`
- `CLOUDINARY_API_KEY`
- `CLOUDINARY_API_SECRET`

## Run

Run the full project from the repository root with the main entrypoint:

```powershell
python main.py
```

Use raw prescription text to run the NER-to-WhatsApp flow:

```powershell
python main.py --text "Patient has viral fever. Give Paracetamol 500 mg twice daily for 5 days."
```

Generate multilingual images only, without sending to WhatsApp:

```powershell
python main.py --generate-only
```

## Notes

- The default demo data in `main.py` is useful for smoke testing the full pipeline.
- The WhatsApp sender uploads local images to Cloudinary first because Twilio requires publicly accessible media URLs.
- Generated language-specific images are written to `generated_prescriptions/`.
