# Prescripto

Prescripto is a multilingual prescription workflow that turns structured or free-text medical input into prescription images in English, Hindi, and Marathi. It can also send those images to a patient over WhatsApp.

## What It Does

The project is designed to make clinic prescriptions easier to read and share. It extracts medicine and diagnosis details from text, translates the structured fields, renders clean prescription images, and optionally delivers them through WhatsApp.

## How It Works

1. A BiLSTM-CRF NER model extracts entities from raw prescription text.
2. The extracted entities are grouped into structured prescription data.
3. Structured text fields are translated into Hindi and Marathi with IndicTrans2.
4. Prescription images are rendered with Pillow.
5. Images are uploaded to Cloudinary.
6. WhatsApp delivery happens through Twilio.

## Project Layout

- `main.py` - command-line entry point for the full pipeline
- `app/gradio_app.py` - Gradio frontend
- `src/send_multilingual_prescription.py` - end-to-end orchestration helper
- `src/ner/` - NER inference, preprocessing, training, and evaluation
- `src/translation/` - translation helpers
- `src/prescription_generator/` - prescription image renderer
- `src/whatsapp/` - Cloudinary upload and Twilio WhatsApp sender
- `models/ner_model/` - trained NER model artifacts
- `generated_prescriptions/` - generated prescription images and records

## Setup

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The full pipeline also needs the runtime libraries used by the source modules, including TensorFlow, PyTorch, Transformers, IndicTransToolkit, Twilio, Cloudinary, and Pillow.

Before sending WhatsApp messages, set these environment variables:

- `TWILIO_ACCOUNT_SID`
- `TWILIO_AUTH_TOKEN`
- `CLOUDINARY_CLOUD_NAME`
- `CLOUDINARY_API_KEY`
- `CLOUDINARY_API_SECRET`

## Run The Project

Run the default pipeline from the repository root:

```powershell
python main.py
```

Use raw prescription text to extract entities first:

```powershell
python main.py --text "Patient has viral fever. Give Paracetamol 500 mg twice daily for 5 days."
```

Generate only the prescription images without sending them to WhatsApp:

```powershell
python main.py --generate-only
```

Run the Gradio frontend:

```powershell
python app/gradio_app.py
```

## Gradio UI (Current)

The current Gradio frontend is a single doctor form:

- Input fields: patient details, symptoms/diagnosis text, prescription text, and WhatsApp number.
- Optional fields: doctor/clinic details in a collapsible section.
- One-click flow: NER extraction -> translation -> image generation -> WhatsApp send.
- Output: a final status message only (no intermediate entity/JSON preview in the UI).

The Gradio app requires all Twilio and Cloudinary environment variables because it performs WhatsApp delivery in its current flow.

## Notes

- The demo data in `main.py` is useful for a quick smoke test.
- Twilio requires public media URLs, which is why the project uploads images to Cloudinary first.
- Generated language-specific images are written to `generated_prescriptions/`.
- Gradio-generated run folders are created under `generated_prescriptions/records/`.
