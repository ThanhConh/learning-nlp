"""
classification_head.py

Pipeline:
Text -> Tokenizer -> Input IDs / Attention Mask
     -> Pretrained Transformer
     -> Contextual representations
     -> [CLS] representation
     -> Task-specific Linear layer
     -> Class logits -> Softmax -> Prediction

Dataset CSV must contain:
    text  : input sentence/review
    label : integer class ID (e.g. 0 or 1)

Install:
    pip install torch transformers pandas scikit-learn
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------

MODEL_NAME = "huawei-noah/TinyBERT_General_4L_312D"
MAX_LENGTH = 128
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 2e-5
NUM_LABELS = 2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ---------------------------------------------------------
# 2. Dataset and Tokenization
# ---------------------------------------------------------

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])
        label = int(self.labels[index])

        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


# ---------------------------------------------------------
# 3. Pretrained Transformer + Classification Head
# ---------------------------------------------------------

class TransformerClassifier(nn.Module):
    def __init__(
        self,
        model_name=MODEL_NAME,
        num_labels=NUM_LABELS,
        freeze_encoder=False
    ):
        super().__init__()

        # Pretrained language representation model
        self.encoder = AutoModel.from_pretrained(model_name)

        hidden_size = self.encoder.config.hidden_size

        # Task-specific classification head.
        # Its parameters are newly initialized for this task.
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Optional: freeze the pretrained encoder.
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask):
        # Step 1: Extract contextual representations
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Shape: [batch_size, sequence_length, hidden_size]
        contextual_representations = outputs.last_hidden_state

        # Step 2: Get the [CLS] representation
        # Shape: [batch_size, hidden_size]
        cls_representation = contextual_representations[:, 0, :]

        # Step 3: Classification head produces class logits
        # Shape: [batch_size, num_labels]
        logits = self.classifier(cls_representation)

        # Step 4: Convert logits into probabilities
        probabilities = torch.softmax(logits, dim=1)

        # Step 5: Select the class with the highest probability
        predictions = torch.argmax(probabilities, dim=1)

        return {
            "contextual_representations": contextual_representations,
            "cls_representation": cls_representation,
            "logits": logits,
            "probabilities": probabilities,
            "predictions": predictions
        }


# ---------------------------------------------------------
# 4. Training
# ---------------------------------------------------------

def train_model(model, train_loader, epochs, learning_rate):
    model.to(DEVICE)

    # CrossEntropyLoss expects raw logits, not softmax probabilities.
    criterion = nn.CrossEntropyLoss()

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=learning_rate
    )

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            loss = criterion(outputs["logits"], labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / max(len(train_loader), 1)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"| Training Loss: {average_loss:.4f}"
        )

    training_time = time.time() - start_time
    return model, training_time


# ---------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------

def evaluate_model(model, data_loader):
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            all_predictions.extend(
                outputs["predictions"].cpu().tolist()
            )
            all_labels.extend(labels.cpu().tolist())

    return {
        "accuracy": accuracy_score(all_labels, all_predictions),
        "f1": f1_score(
            all_labels,
            all_predictions,
            average="binary",
            zero_division=0
        )
    }


# ---------------------------------------------------------
# 6. Inference: predict a single text
# ---------------------------------------------------------

def predict_text(text, model, tokenizer, max_length=MAX_LENGTH):
    model.eval()

    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    input_ids = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    probabilities = outputs["probabilities"][0].cpu()
    prediction = int(outputs["predictions"][0].cpu())

    return {
        "text": text,
        "input_ids": input_ids[0].cpu().tolist(),
        "attention_mask": attention_mask[0].cpu().tolist(),
        "cls_representation_shape": tuple(
            outputs["cls_representation"].shape
        ),
        "logits": outputs["logits"][0].cpu().tolist(),
        "probabilities": probabilities.tolist(),
        "prediction": prediction
    }


# ---------------------------------------------------------
# 7. Main: load CSV, train, evaluate, save, and predict
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="CSV file containing text and label columns"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="tinybert_classifier.pt"
    )
    args = parser.parse_args()

    data = pd.read_csv(args.csv)

    if not {"text", "label"}.issubset(data.columns):
        raise ValueError(
            "CSV must contain columns named 'text' and 'label'."
        )

    data = data.dropna(subset=["text", "label"]).copy()
    data["label"] = data["label"].astype(int)

    train_texts, val_texts, train_labels, val_labels = (
        train_test_split(
            data["text"],
            data["label"],
            test_size=0.2,
            random_state=42,
            stratify=data["label"]
        )
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = TextClassificationDataset(
        train_texts,
        train_labels,
        tokenizer,
        MAX_LENGTH
    )

    val_dataset = TextClassificationDataset(
        val_texts,
        val_labels,
        tokenizer,
        MAX_LENGTH
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = TransformerClassifier(
        model_name=MODEL_NAME,
        num_labels=NUM_LABELS,
        freeze_encoder=False
    )

    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(
        "Total parameters:",
        sum(p.numel() for p in model.parameters())
    )
    print(
        "Trainable parameters:",
        sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )
    )

    model, training_time = train_model(
        model,
        train_loader,
        EPOCHS,
        LEARNING_RATE
    )

    metrics = evaluate_model(model, val_loader)

    print(f"Validation Accuracy: {metrics['accuracy']:.4f}")
    print(f"Validation F1: {metrics['f1']:.4f}")
    print(f"Training time: {training_time:.2f}s")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": MODEL_NAME,
            "num_labels": NUM_LABELS,
            "max_length": MAX_LENGTH
        },
        args.checkpoint
    )
    print(f"Saved checkpoint: {args.checkpoint}")

    # Example inference
    sample_text = (
        "This movie was excellent. "
        "The story and acting were wonderful."
    )

    result = predict_text(
        sample_text,
        model,
        tokenizer,
        MAX_LENGTH
    )

    print("\nSingle-text prediction")
    print("Text:", result["text"])
    print("Input IDs:", result["input_ids"][:20], "...")
    print(
        "Attention mask:",
        result["attention_mask"][:20],
        "..."
    )
    print(
        "CLS representation shape:",
        result["cls_representation_shape"]
    )
    print("Logits:", result["logits"])
    print("Probabilities:", result["probabilities"])
    print("Predicted class:", result["prediction"])


if __name__ == "__main__":
    main()
