# Bertsonify

**Bertsonify** is a lightweight, extendable Python library for training and using BERT-style models for Named Entity Recognition (NER) based on structured JSON data.

## Features

- Uses any Hugging Face BERT-compatible model (like `distilbert`, `bert-base`, etc.)
- Converts `(prompt, structured_output)` JSON into token-level labels (BIO format).
- Handles repeated values and multiple entities in one sentence.
- Returns output in the same structured format as the input.

## Installation (Optional)

To install locally as a package:

```bash
pip install -e .
```

##Sample usage
```
from bertsonify.core import GenericNER

# Sample training data: one prompt, one structured JSON output
samples = [
    {
        "input": "Искам две бири и две води.",
        "output": '{"items":[{"quantity":"две","name":"бира"},{"quantity":"две","name":"вода"}]}'
    }
]

# Create a GenericNER instance
ner = GenericNER(
    model_name="distilbert-base-multilingual-cased",
    tag_field_map={"B-QUANTITY": "quantity", "B-ITEM": "name"}  # map tags to JSON keys
)

# Train the model on the small dataset
ner.train(samples)

# Predict using trained model
result = ner.predict("Ще взема три бири и една вода.")

# Use the result programmatically
for item in result["items"]:
    print(f"{item.get('quantity', '?')} x {item.get('name', '?')}")
```
