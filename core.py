import torch
import json
import os
import re
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer
)


class GenericNER:
    def __init__(self, model_name, model_dir="./ner-model", tag_field_map=None, data_cache_dir="./ner-data"):
        self.model_name = model_name
        self.model_dir = model_dir
        self.data_cache_dir = data_cache_dir
        self.tag_field_map = tag_field_map or {"B-QUANTITY": "quantity", "B-ITEM": "name"}
        self.tags = ["O"] + list(self.tag_field_map.keys())
        self.label2id = {label: i for i, label in enumerate(self.tags)}
        self.id2label = {i: label for label, i in self.label2id.items()}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None

    def _convert_json_to_token_labels(self, sample):
        text = sample["input"]
        data = json.loads(sample["output"])

        # Parse item list
        if isinstance(data, list):
            entities = data
        elif isinstance(data, dict) and "items" in data:
            entities = data["items"]
        else:
            entities = []

        words = text.split()
        tokens = []
        labels = []

        word_spans = []
        idx = 0
        for word in words:
            start = text.find(word, idx)
            end = start + len(word)
            word_spans.append((start, end))
            idx = end

        token_word_map = []
        for i, word in enumerate(words):
            word_tokens = self.tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            token_word_map.extend([i] * len(word_tokens))

        labels_per_word = ["O"] * len(words)

        lowered_text = text.lower()
        for item in entities:
            for tag, field in self.tag_field_map.items():
                val = str(item.get(field, "")).strip().lower()
                if not val:
                    continue

                # Find all occurrences of val in text
                for match in re.finditer(re.escape(val), lowered_text):
                    start, end = match.start(), match.end()
                    for i, (w_start, w_end) in enumerate(word_spans):
                        if w_start >= start and w_end <= end:
                            labels_per_word[i] = tag

        labels = [self.label2id.get(labels_per_word[word_idx], 0) for word_idx in token_word_map]

        print("\n[Sample] Input:", text)
        print("[Sample] Output JSON:", json.dumps(data, ensure_ascii=False))
        print("[Sample] Tokens:", tokens)
        print("[Sample] Labels:", [self.id2label.get(l, 'O') for l in labels])

        return {"tokens": tokens, "ner_tags": [self.id2label.get(l, 'O') for l in labels]}

    def _tokenize_and_align_labels(self, example):
        tokenized_inputs = self.tokenizer(
            example["tokens"],
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=64
        )
        word_ids = tokenized_inputs.word_ids()
        labels = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                labels.append(-100)
            elif word_idx != previous_word_idx:
                labels.append(self.label2id[example["ner_tags"][word_idx]])
            else:
                labels.append(-100)
            previous_word_idx = word_idx
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def load_training_file(self, file_path):
        with open(file_path, encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines()]
        assert len(lines) % 2 == 0, "Training file must have even number of lines (input/output pairs)."
        print(f"[Loader] Loaded {len(lines)//2} pairs from {file_path}")
        return [{"input": lines[i], "output": lines[i+1]} for i in range(0, len(lines), 2)]

    def train(self, json_samples=None, file_path=None, epochs=5):
        os.makedirs(self.data_cache_dir, exist_ok=True)

        if file_path:
            json_samples = self.load_training_file(file_path)

        print(f"[Trainer] Preparing {len(json_samples)} samples...")
        structured = [self._convert_json_to_token_labels(s) for s in json_samples]
        dataset = Dataset.from_list(structured)
        tokenized_dataset = dataset.map(self._tokenize_and_align_labels)

        print(f"[Trainer] Saving tokenized dataset to {self.data_cache_dir}")
        tokenized_dataset.save_to_disk(self.data_cache_dir)

        print(f"[Trainer] Initializing model {self.model_name}")
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name, num_labels=len(self.tags)
        )

        args = TrainingArguments(
            output_dir=self.model_dir,
            per_device_train_batch_size=4,
            num_train_epochs=epochs,
            logging_dir=f"{self.model_dir}/logs",
            save_strategy="epoch"
        )

        print("[Trainer] Starting training...")
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=tokenized_dataset
        )
        trainer.train()

        print(f"[Trainer] Saving model to {self.model_dir}")
        self.model.save_pretrained(self.model_dir)
        self.tokenizer.save_pretrained(self.model_dir)

    def load(self):
        print(f"[Model] Loading from {self.model_dir}")
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

    def load_from_directory(self, path):
        self.model_dir = path
        self.load()

    def get_model(self):
        if self.model is None:
            self.load()
        return self.model

    def predict(self, text):
        if self.model is None:
            self.load()

        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits

        predictions = torch.argmax(logits, dim=2)[0].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        word_ids = inputs.word_ids()
        word_tags = {}

        for token, prediction, word_idx in zip(tokens, predictions, word_ids):
            if word_idx is not None:
                label = self.id2label[prediction]
                if label != "O":
                    word_tags[word_idx] = label

        result = {"results": []}
        current_item = {}
        used_fields = set()

        for word_idx, word in enumerate(text.split()):
            label = word_tags.get(word_idx, "O")
            if label != "O":
                field = self.tag_field_map.get(label)
                if field:
                    current_item[field] = word
                    used_fields.add(field)

                # Once we've collected a full item worth of fields, store and reset
                if used_fields == set(self.tag_field_map.values()):
                    result["results"].append(current_item)
                    current_item = {}
                    used_fields = set()

        if current_item:
            result["results"].append(current_item)

        return result