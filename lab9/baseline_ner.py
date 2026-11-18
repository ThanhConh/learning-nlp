"""
Core implementation of Named Entity Recognition (NER).

Pipeline:
    Words
    -> Tokenization
    -> word_ids
    -> Alignment
    -> token_labels
    -> Model Forward
    -> logits
    -> predictions
    -> NER results
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
)


# ============================================================
# 1. Configuration
# ============================================================

MODEL_NAME = "dslim/bert-base-NER"

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# Ignore special tokens and non-first subword tokens
IGNORE_INDEX = -100


# ============================================================
# 2. Load tokenizer and model
# ============================================================

def load_model(model_name=MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = AutoModelForTokenClassification.from_pretrained(
        model_name
    )

    model.to(DEVICE)
    model.eval()

    return tokenizer, model


# ============================================================
# 3. Tokenization
# ============================================================

def tokenize_words(tokenizer, words):
    """
    Tokenize a list of pre-split words.

    Args:
        tokenizer: Hugging Face tokenizer.
        words: List of words.

    Returns:
        tokens: Tokenizer output.
    """

    tokens = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )

    return tokens


# ============================================================
# 4. Word IDs
# ============================================================

def get_word_ids(tokens, batch_index=0):
    """
    Get word_ids mapping between subwords and original words.

    None indicates a special token.
    """

    word_ids = tokens.word_ids(
        batch_index=batch_index
    )

    return word_ids


# ============================================================
# 5. Label Alignment
# ============================================================

def align_labels(
    word_ids,
    word_labels,
    label2id,
):
    """
    Align word-level labels with subword tokens.

    Rules:
        - Special tokens -> -100
        - First subword -> corresponding word label
        - Remaining subwords -> -100

    Args:
        word_ids: Mapping from subwords to original words.
        word_labels: Labels at word-level.
        label2id: Mapping from label name to label ID.

    Returns:
        token_labels: Labels aligned with subwords.
    """

    token_labels = []

    previous_word_id = None

    for word_id in word_ids:

        # Special tokens
        if word_id is None:
            token_labels.append(IGNORE_INDEX)

        # First subword of a word
        elif word_id != previous_word_id:

            label = word_labels[word_id]

            token_labels.append(
                label2id[label]
            )

        # Remaining subwords
        else:
            token_labels.append(IGNORE_INDEX)

        previous_word_id = word_id

    return token_labels


# ============================================================
# 6. Model Forward
# ============================================================

def forward_model(
    model,
    input_ids,
    attention_mask,
    token_labels=None,
):
    """
    Run Transformer forward pass.

    Returns:
        logits
        loss (if labels are provided)
    """

    input_ids = input_ids.to(DEVICE)

    attention_mask = attention_mask.to(DEVICE)

    if token_labels is not None:

        token_labels = torch.tensor(
            token_labels,
            dtype=torch.long,
        ).unsqueeze(0)

        token_labels = token_labels.to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=token_labels,
        )

        return outputs.logits, outputs.loss

    else:

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        return outputs.logits, None


# ============================================================
# 7. Predictions
# ============================================================

def get_predictions(logits):
    """
    Convert logits into predicted label IDs.
    """

    predictions = torch.argmax(
        logits,
        dim=-1,
    )

    return predictions


# ============================================================
# 8. Decode Predictions
# ============================================================

def decode_predictions(
    words,
    word_ids,
    predictions,
    id2label,
):
    """
    Convert subword predictions into word-level predictions.

    Only the first subword of each word is used.
    """

    predictions = predictions[0].tolist()

    word_predictions = []

    previous_word_id = None

    for token_index, word_id in enumerate(word_ids):

        # Ignore special tokens
        if word_id is None:
            continue

        # Ignore duplicated subwords
        if word_id == previous_word_id:
            continue

        label_id = predictions[token_index]

        label = id2label[label_id]

        word_predictions.append(
            (words[word_id], label)
        )

        previous_word_id = word_id

    return word_predictions


# ============================================================
# 9. Display Pipeline
# ============================================================

def display_pipeline(
    words,
    tokens,
    word_ids,
    token_labels,
    logits,
    predictions,
    word_predictions,
    id2label,
):
    """
    Display the intermediate outputs of the NER pipeline.
    """

    print("\n" + "=" * 60)
    print("NER CORE IMPLEMENTATION")
    print("=" * 60)

    # Original words
    print("\n[1] Original words:")
    print(words)

    # Tokenization
    print("\n[2] Tokens:")
    print(
        tokens.tokens()
    )

    # Input IDs
    print("\n[3] Input IDs:")
    print(
        tokens["input_ids"][0].tolist()
    )

    # Attention mask
    print("\n[4] Attention Mask:")
    print(
        tokens["attention_mask"][0].tolist()
    )

    # Word IDs
    print("\n[5] Word IDs:")
    print(word_ids)

    # Labels
    print("\n[6] Token Labels:")
    print(token_labels)

    # Logits
    print("\n[7] Logits shape:")
    print(
        tuple(logits.shape)
    )

    # Predictions
    print("\n[8] Predictions:")
    print(
        predictions[0].tolist()
    )

    # Prediction labels
    print("\n[9] Predicted Labels:")

    for word, label in word_predictions:
        print(
            f"{word:15s} -> {label}"
        )


# ============================================================
# 10. Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Example sentence
    # --------------------------------------------------------

    words = [
        "Barack",
        "Obama",
        "was",
        "born",
        "in",
        "Hawaii",
        ".",
    ]

    # Word-level ground truth labels
    word_labels = [
        "B-PER",
        "I-PER",
        "O",
        "O",
        "O",
        "B-LOC",
        "O",
    ]

    # --------------------------------------------------------
    # Load model and tokenizer
    # --------------------------------------------------------

    tokenizer, model = load_model()

    label2id = model.config.label2id

    id2label = model.config.id2label

    print("Model labels:")
    print(id2label)

    # --------------------------------------------------------
    # Tokenization
    # --------------------------------------------------------

    tokens = tokenize_words(
        tokenizer,
        words,
    )

    # --------------------------------------------------------
    # Word IDs
    # --------------------------------------------------------

    word_ids = get_word_ids(
        tokens,
        batch_index=0,
    )

    # --------------------------------------------------------
    # Alignment
    # --------------------------------------------------------

    token_labels = align_labels(
        word_ids,
        word_labels,
        label2id,
    )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    logits, loss = forward_model(
        model=model,
        input_ids=tokens["input_ids"],
        attention_mask=tokens["attention_mask"],
        token_labels=token_labels,
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions = get_predictions(
        logits
    )

    # --------------------------------------------------------
    # Decode
    # --------------------------------------------------------

    word_predictions = decode_predictions(
        words=words,
        word_ids=word_ids,
        predictions=predictions,
        id2label=id2label,
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    display_pipeline(
        words=words,
        tokens=tokens,
        word_ids=word_ids,
        token_labels=token_labels,
        logits=logits,
        predictions=predictions,
        word_predictions=word_predictions,
        id2label=id2label,
    )

    if loss is not None:

        print("\n[10] Loss:")
        print(
            loss.item()
        )


if __name__ == "__main__":
    main()