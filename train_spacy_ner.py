
import spacy
import json
from pathlib import Path
import random
from spacy.training.example import Example

# 🔁 Load training data
with open(r"C:\Users\Gouthum\Downloads\Project\batch_1\batch_1\MLDL\train_test_validation_spacy\train_data.json", "r", encoding="utf-8") as f:
    TRAIN_DATA = json.load(f)

with open(r"C:\Users\Gouthum\Downloads\Project\batch_1\batch_1\MLDL\train_test_validation_spacy\val_data.json", "r", encoding="utf-8") as f:
    VAL_DATA = json.load(f)

# 🧠 Create blank model and add NER pipe
nlp = spacy.blank("en")
ner = nlp.add_pipe("ner")

# 🏷 Add labels to the NER component
for _, annotations in TRAIN_DATA:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

# ✅ Initialize model
nlp.initialize()

# 🎯 Training loop
n_iter = 15
for itn in range(n_iter):
    print(f"🔁 Iteration {itn+1}/{n_iter}")
    random.shuffle(TRAIN_DATA)
    losses = {}
    for text, annotations in TRAIN_DATA:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        nlp.update([example], losses=losses)
    print("Losses:", losses)

# 💾 Save model
output_dir = Path(r"C:\Users\Gouthum\Downloads\Project\batch_1\batch_1\MLDL\spacy_ner_model")
nlp.to_disk(output_dir)
print(f"✅ Model saved to {output_dir}")
