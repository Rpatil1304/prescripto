## 📋 Project Overview

**AI-Powered Multilingual Prescription Generator & WhatsApp Delivery System**

A deep learning-based clinical workflow tool that lets doctors generate and deliver
digital prescriptions to patients instantly via WhatsApp — in English, Hindi, and Marathi.

### The Problem
Prescriptions in Indian clinics are still largely handwritten, illegible, and
language-inaccessible for many patients. There's no standardized digital
record the patient can refer back to, share with a pharmacist, or read in
their native language.

### The Solution
This system lets a doctor fill a simple form (patient name, symptoms,
prescription text, and phone number) and, in a single click:

1. **Extracts structured medical entities** (drug names, dosages, frequency,
   duration, symptoms) from the doctor's free-text input using a custom-trained
   **BiLSTM-CRF Named Entity Recognition (NER) model**
2. **Translates** the structured prescription into Hindi and Marathi using
   **AI4Bharat's IndicTrans2** transformer model
3. **Generates clean, professional prescription images** (PNG) in all three
   languages using a templated rendering pipeline
4. **Delivers the prescriptions directly to the patient's WhatsApp** via the
   Twilio WhatsApp Business API — no app installation or manual paperwork required

### Key Features
- 🧠 Custom NER model trained on real (MTSamples) and synthetic clinical text
- 🌐 Multilingual output — English, Hindi, Marathi — from a single input
- 🖼️ Auto-generated, print-ready prescription images
- 📲 One-click WhatsApp delivery to the patient
- 📊 Experiment tracking via MLflow for model iteration
- 🖥️ Simple Streamlit interface — no technical skill required for doctors

### Tech Stack
| Component | Technology |
|---|---|
| NER Model | TensorFlow/Keras (BiLSTM-CRF) |
| Translation | AI4Bharat IndicTrans2 |
| Image Generation | Python Pillow |
| Messaging | Twilio WhatsApp API |
| Interface | Streamlit |
| Experiment Tracking | MLflow |
| Annotation | Label Studio (BIO tagging) |

### Why This Project
Built to demonstrate an end-to-end applied ML pipeline — from raw clinical
text and manual annotation, through model training and evaluation, to a
real-world deployable product with third-party API integration — rather
than an isolated notebook experiment.