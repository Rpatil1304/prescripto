"""
ner_inference.py
------------------
Loads the trained BiLSTM-CRF model and provides two functions:

1. extract_entities(text) -> list of (entity_type, entity_text) tuples,
   by tokenizing raw free-text input, running it through the model, and
   decoding the predicted BIO tags back into entity spans.

2. group_into_prescription_data(entities, patient_name=..., date=..., ...)
   -> a dict in exactly the format `prescription_generator.py` expects
   (medicines list, diagnosis list, symptoms list), by grouping
   consecutive DRUG/DOSAGE/FREQUENCY/DURATION tags into per-medicine
   entries.

This is the connector between the trained NER model (Phase 2) and the
prescription image generator (Phase 3).
"""

import re
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from crf_layer import CRF, crf_decode


# ---------------------------------------------------------------------------
# Same tokenizer used during BIO dataset conversion (Phase 1) — must match
# exactly, or token boundaries at inference time won't line up with what
# the model was trained on.
# ---------------------------------------------------------------------------

TOKEN_RE = re.compile(r"\w+(?:\.\w+)*|[^\w\s]")


def tokenize(sentence):
    return [(m.start(), m.end(), m.group()) for m in TOKEN_RE.finditer(sentence)]


class NERModel:
    """Wraps the trained BiLSTM-CRF model + vocabs for easy inference."""

    def __init__(self, model_dir):
        with open(f"{model_dir}/word_vocab.json") as f:
            self.word_vocab = json.load(f)
        with open(f"{model_dir}/char_vocab.json") as f:
            self.char_vocab = json.load(f)
        with open(f"{model_dir}/tag_vocab.json") as f:
            self.tag_vocab = json.load(f)
        with open(f"{model_dir}/preprocessing_config.json") as f:
            self.config = json.load(f)

        self.inv_tag_vocab = {v: k for k, v in self.tag_vocab.items()}
        self.max_len = self.config["MAX_LEN"]
        self.max_word_len = self.config["MAX_WORD_LEN"]

        self.base_model = load_model(f"{model_dir}/bilstm_encoder.keras")

        num_tags = self.config["tag_vocab_size"]
        self.crf = CRF(num_tags=num_tags)
        self.crf.build((None, self.max_len, num_tags))
        transitions = np.load(f"{model_dir}/crf_transitions.npy")
        self.crf.transitions.assign(transitions)

    def _encode_words(self, tokens):
        unk = self.word_vocab["<UNK>"]
        pad = self.word_vocab["<PAD>"]
        ids = [self.word_vocab.get(t.lower(), unk) for t in tokens][: self.max_len]
        ids += [pad] * (self.max_len - len(ids))
        return ids

    def _encode_chars(self, tokens):
        unk = self.char_vocab["<UNK>"]
        pad = self.char_vocab["<PAD>"]
        char_ids = []
        for tok in tokens[: self.max_len]:
            c_ids = [self.char_vocab.get(ch, unk) for ch in tok][: self.max_word_len]
            c_ids += [pad] * (self.max_word_len - len(c_ids))
            char_ids.append(c_ids)
        while len(char_ids) < self.max_len:
            char_ids.append([pad] * self.max_word_len)
        return char_ids

    def extract_entities(self, text):
        """
        Returns a list of (entity_type, entity_text) tuples found in `text`.
        E.g. [("DRUG", "Paracetamol"), ("DOSAGE", "500 mg"), ...]
        Preserves the order entities appear in the input text.
        """
        token_spans = tokenize(text)
        if not token_spans:
            return []

        tokens = [t[2] for t in token_spans]
        real_len = min(len(tokens), self.max_len)

        word_ids = self._encode_words(tokens)
        char_ids = self._encode_chars(tokens)

        word_batch = tf.constant([word_ids])
        char_batch = tf.constant([char_ids])
        seq_len_batch = tf.constant([real_len], dtype=tf.int32)

        potentials = self.base_model([word_batch, char_batch], training=False)
        predicted_tags = self.crf(potentials, seq_len_batch).numpy()[0]

        # Decode BIO tags back into entity spans
        entities = []
        current_type = None
        current_tokens = []

        for i in range(real_len):
            tag = self.inv_tag_vocab.get(int(predicted_tags[i]), "O")
            if tag.startswith("B-"):
                if current_type is not None:
                    entities.append((current_type, " ".join(current_tokens)))
                current_type = tag[2:]
                current_tokens = [tokens[i]]
            elif tag.startswith("I-") and current_type == tag[2:]:
                current_tokens.append(tokens[i])
            else:
                if current_type is not None:
                    entities.append((current_type, " ".join(current_tokens)))
                current_type = None
                current_tokens = []

        if current_type is not None:
            entities.append((current_type, " ".join(current_tokens)))

        return entities


def group_into_prescription_data(entities, patient_name="", date="", age="",
                                  clinic_name="", doctor_name="", clinic_address=""):
    """
    Groups a flat list of (entity_type, text) tuples into the structured
    dict format `prescription_generator.generate_prescription_image()`
    expects.

    Grouping logic: a new medicine entry starts at each DRUG tag. Any
    DOSAGE / FREQUENCY / DURATION tags that follow (before the next DRUG)
    are attached to that medicine. DIAGNOSIS and SYMPTOM tags are
    collected into their own separate lists, since they describe the
    patient's condition rather than belonging to a specific medicine.

    This grouping is entirely driven by tag order/adjacency in the
    model's output — no fixed number of medicines or fixed slots assumed,
    so it naturally handles 1 medicine or 10 medicines the same way.
    """
    medicines = []
    diagnosis = []
    symptoms = []
    current_medicine = None

    for entity_type, text in entities:
        if entity_type == "DRUG":
            if current_medicine is not None:
                medicines.append(current_medicine)
            current_medicine = {"drug": text, "dosage": "", "frequency": "", "duration": ""}
        elif entity_type == "DOSAGE" and current_medicine is not None:
            current_medicine["dosage"] = text
        elif entity_type == "FREQUENCY" and current_medicine is not None:
            current_medicine["frequency"] = text
        elif entity_type == "DURATION" and current_medicine is not None:
            current_medicine["duration"] = text
        elif entity_type == "DIAGNOSIS":
            diagnosis.append(text)
        elif entity_type == "SYMPTOM":
            symptoms.append(text)
        # any DOSAGE/FREQUENCY/DURATION appearing before the first DRUG is
        # dropped rather than crashing — defensive handling of unexpected
        # model output order, since real-world free text won't always be
        # perfectly structured

    if current_medicine is not None:
        medicines.append(current_medicine)

    return {
        "clinic_name": clinic_name,
        "doctor_name": doctor_name,
        "clinic_address": clinic_address,
        "patient_name": patient_name,
        "date": date,
        "age": age,
        "diagnosis": diagnosis,
        "symptoms": symptoms,
        "medicines": medicines,
    }


if __name__ == "__main__":
    MODEL_DIR = "models/ner_model"

    print("Loading model...")
    ner = NERModel(MODEL_DIR)

    test_sentences = [
        "Patient has viral fever. Give Paracetamol 500 mg twice daily for 5 days.",
        "Metformin 500 mg three times a day, Lisinopril 10 mg daily, aspirin 81 mg daily.",
    ]

    for sentence in test_sentences:
        print(f"\nInput: {sentence}")
        entities = ner.extract_entities(sentence)
        print("Extracted entities:", entities)

        structured = group_into_prescription_data(
            entities, patient_name="Test Patient", date="11 July 2026"
        )
        print("Grouped prescription data:", json.dumps(structured, indent=2))
