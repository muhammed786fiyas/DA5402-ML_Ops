import spacy
import os
import json
from datetime import datetime

# Load model once
nlp = spacy.load("en_core_web_sm")

# Ensure output directory exists
NER_OUTPUT_DIR = "app/outputs/ner"
os.makedirs(NER_OUTPUT_DIR, exist_ok=True)

def extract_entities(text: str):
    doc = nlp(text)

    entities = [
        {
            "text": ent.text,
            "label": ent.label_
        }
        for ent in doc.ents
    ]

    # Save output as JSON file
    filename = f"ner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(NER_OUTPUT_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(entities, f, indent=4)

    return entities