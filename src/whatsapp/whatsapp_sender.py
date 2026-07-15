"""
whatsapp_sender.py
-------------------
Sends generated prescription images to a patient's WhatsApp number using
the Twilio WhatsApp API.

IMPORTANT CONSTRAINT (read before using): Twilio requires media to be at a
PUBLICLY ACCESSIBLE URL — it cannot accept a local file path directly. Your
prescription images are generated locally, so they need to be hosted
publicly first.

This module uses Cloudinary (a real, established cloud media hosting
service with a generous free tier) to upload each image and get a public
URL back, right before sending it. Earlier attempts with free anonymous
upload sites (catbox.moe, 0x0.st) proved unreliable — inconsistent bot
blocking and downtime — so this project uses a proper hosting service
instead, which is both more reliable and a more defensible engineering
choice for a real project.

NOTE: even Cloudinary's free tier serves images at public URLs (anyone
with the exact URL can view them) — fine for a personal project's
development and demo needs with test data, but not appropriate for real
patient information without additional access controls (e.g. signed,
expiring URLs, which Cloudinary also supports on paid tiers).

Setup required before this will work:
    1. Free Cloudinary account (cloudinary.com) — no credit card required
    2. From your Cloudinary dashboard, copy: Cloud Name, API Key, API Secret
    3. Free Twilio account (twilio.com)
    4. Activate the WhatsApp Sandbox (Twilio Console -> Messaging -> Try it
       out -> Send a WhatsApp message)
    5. From your own WhatsApp, send "join <your-sandbox-keyword>" to the
       sandbox number shown in the console — this opts your number in
    6. Set environment variables (do NOT hardcode credentials in code):
           TWILIO_ACCOUNT_SID
           TWILIO_AUTH_TOKEN
           CLOUDINARY_CLOUD_NAME
           CLOUDINARY_API_KEY
           CLOUDINARY_API_SECRET
"""

import os
import time
from pathlib import Path
import cloudinary
import cloudinary.uploader
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException


_cloudinary_configured = False


def _ensure_cloudinary_configured():
    """Configures the Cloudinary SDK once, reading credentials from
    environment variables. Safe to call repeatedly — only configures once."""
    global _cloudinary_configured
    if _cloudinary_configured:
        return

    cloud_name = os.environ.get("CLOUDINARY_CLOUD_NAME")
    api_key = os.environ.get("CLOUDINARY_API_KEY")
    api_secret = os.environ.get("CLOUDINARY_API_SECRET")

    if not all([cloud_name, api_key, api_secret]):
        raise ValueError(
            "Cloudinary credentials not found. Set CLOUDINARY_CLOUD_NAME, "
            "CLOUDINARY_API_KEY, and CLOUDINARY_API_SECRET as environment "
            "variables (get these free from your Cloudinary dashboard)."
        )

    cloudinary.config(cloud_name=cloud_name, api_key=api_key, api_secret=api_secret, secure=True)
    _cloudinary_configured = True


def upload_to_cloudinary(file_path):
    """
    Uploads a local image file to Cloudinary and returns a public HTTPS URL.
    Requires a free Cloudinary account (see module docstring for setup).
    """
    _ensure_cloudinary_configured()
    result = cloudinary.uploader.upload(file_path)
    url = result.get("secure_url")
    if not url:
        raise RuntimeError(f"Cloudinary upload did not return a URL. Response: {result}")
    return url


# Twilio's shared WhatsApp Sandbox number — same for everyone during
# development, until you get a dedicated approved WhatsApp sender.
SANDBOX_FROM_NUMBER = "whatsapp:+14155238886"


class WhatsAppSender:
    def __init__(self, account_sid=None, auth_token=None, from_number=SANDBOX_FROM_NUMBER):
        account_sid = account_sid or os.environ.get("TWILIO_ACCOUNT_SID")
        auth_token = auth_token or os.environ.get("TWILIO_AUTH_TOKEN")

        if not account_sid or not auth_token:
            raise ValueError(
                "Twilio credentials not found. Set TWILIO_ACCOUNT_SID and "
                "TWILIO_AUTH_TOKEN as environment variables, or pass them "
                "directly to WhatsAppSender()."
            )

        self.client = Client(account_sid, auth_token)
        self.from_number = from_number

    def send_image(self, to_phone_number, image_url, caption=""):
        """
        Sends a single image to a WhatsApp number.

        to_phone_number: E.164 format, e.g. "+919876543210" (no "whatsapp:"
            prefix needed here — it's added automatically).
        image_url: a PUBLICLY accessible URL to the image.
        caption: optional text sent alongside the image.

        Returns the Twilio message SID on success, or raises
        TwilioRestException on failure (e.g. invalid number, unopted-in
        sandbox recipient, media fetch failure, 24-hour session window
        expired — error 63016, which needs a fresh "join" message).
        """
        to_whatsapp = f"whatsapp:{to_phone_number}" if not to_phone_number.startswith("whatsapp:") else to_phone_number

        message = self.client.messages.create(
            body=caption,
            media_url=[image_url],
            from_=self.from_number,
            to=to_whatsapp,
        )
        return message.sid

    def send_prescription_images(self, to_phone_number, image_file_paths,
                                  language_labels=None, delay_seconds=1.0):
        """
        Sends multiple prescription images (e.g. English, Hindi, Marathi
        versions) to a patient, one message per image. Each image is
        uploaded to Cloudinary automatically to get a public URL — you
        just pass local file paths.

        image_file_paths: list of local file paths, e.g.
            ["outputs/prescription_en.png", "outputs/prescription_hi.png"]
        language_labels: optional list of captions matching image_file_paths
            by position, e.g. ["English", "Hindi", "Marathi"]. If omitted,
            no caption is sent.
        delay_seconds: small pause between messages — avoids hitting
            Twilio's rate limits when sending several messages back-to-back,
            and keeps messages arriving in a sensible order on WhatsApp.

        Returns a list of dicts: [{"file_path": ..., "sid": ... or None,
        "error": ... or None}] — one entry per image, so a failure on one
        language (upload failure OR send failure) doesn't silently hide
        whether the others succeeded.
        """
        results = []
        for i, file_path in enumerate(image_file_paths):
            caption = ""
            if language_labels and i < len(language_labels):
                caption = f"Prescription ({language_labels[i]})"

            try:
                print(f"  Uploading {file_path} to Cloudinary...")
                image_url = upload_to_cloudinary(file_path)
                print(f"  Upload succeeded: {image_url}")
                sid = self.send_image(to_phone_number, image_url, caption=caption)
                results.append({"file_path": file_path, "sid": sid, "error": None})
            except (TwilioRestException, RuntimeError, ValueError, OSError) as e:
                results.append({"file_path": file_path, "sid": None, "error": str(e)})

            if i < len(image_file_paths) - 1:
                time.sleep(delay_seconds)

        return results


if __name__ == "__main__":
    import sys

    src_root = Path(__file__).resolve().parents[1]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from translation.prescripto_translator import PrescriptionTranslator, LANG_CODES
    from prescription_generator.prescription_generator import generate_prescription_image

    # --- Configuration for a manual test run ---
    # Fill these in with your real values before running.
    PATIENT_PHONE_NUMBER = "+919518398826"       # the number that joined your sandbox
    OUTPUT_DIR = Path("generated_prescriptions")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

    print("Loading translation model (first run downloads ~1GB, cached after)...")
    translator = PrescriptionTranslator()

    print("Translating to Hindi and Marathi...")
    hindi_data = translator.translate_prescription_data(sample_data, LANG_CODES["hindi"])
    marathi_data = translator.translate_prescription_data(sample_data, LANG_CODES["marathi"])

    print("Generating prescription images (English, Hindi, Marathi)...")
    image_paths = []
    language_labels = ["English", "Hindi", "Marathi"]
    for data, label in zip([sample_data, hindi_data, marathi_data], language_labels):
        image = generate_prescription_image(data)
        path = OUTPUT_DIR / f"prescription_{label.lower()}.png"
        image.save(path)
        image_paths.append(str(path))
        print(f"  Saved {label} -> {path}")

    sender = WhatsAppSender()  # reads TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN from env

    print(f"Sending {len(image_paths)} image(s) to {PATIENT_PHONE_NUMBER}...")
    results = sender.send_prescription_images(
        PATIENT_PHONE_NUMBER, image_paths, language_labels
    )

    for r in results:
        if r["error"]:
            print(f"  FAILED  {r['file_path']}: {r['error']}")
        else:
            print(f"  SENT    {r['file_path']} (message SID: {r['sid']})")