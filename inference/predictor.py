"""Standalone inference pipeline for Streamlit deployment.

No dependencies on training infrastructure - everything is self-contained.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# Paths
APP_DIR = Path(__file__).resolve().parent.parent
CKPT_DIR = APP_DIR / "models" / "undersampling_no_environment"
DEVICE = torch.device("cpu")
MAX_LENGTH = 512

# Model specifications: (model_key, hf_name, freeze_until)
MODEL_SPECS = [
    ("bert", "bert-base-multilingual-cased", None),
    ("distilbert", "distilbert-base-multilingual-cased", None),
    ("electra", "google/electra-base-discriminator", None),
    ("roberta", "xlm-roberta-base", None),
]

# Model info
MODEL_INFO: dict[str, dict] = {
    "roberta": {"display": "RoBERTa", "accuracy": 0.9175, "macro_f1": 0.9177},
    "distilbert": {"display": "DistilBERT", "accuracy": 0.9107, "macro_f1": 0.9111},
    "electra": {"display": "ELECTRA", "accuracy": 0.8746, "macro_f1": 0.8761},
    "bert": {"display": "BERT", "accuracy": 0.8418, "macro_f1": 0.8429},
}

DEFAULT_MODEL = "distilbert"

# Class labels for Cambodian news
LABELS = [
    "Politics & Government",
    "Business & Economics",
    "Sports",
    "Entertainment",
    "Technology",
]


def preprocess(text: str) -> str:
    """Simple text preprocessing: lowercase, remove special chars, strip whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = " ".join(text.split())
    return text


class TransformerClassifier(nn.Module):
    """Simple transformer-based classifier."""

    def __init__(self, model_name: str, num_classes: int = 5):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits


@lru_cache(maxsize=1)
def get_labels() -> list[str]:
    """Return ordered class names."""
    return LABELS


def available_models() -> list[str]:
    """Return model keys whose checkpoint file exists."""
    return [key for key, _, _ in MODEL_SPECS if (CKPT_DIR / f"{key}_best.pt").is_file()]


@lru_cache(maxsize=4)
def load_model(model_key: str):
    """Load model and tokenizer."""
    spec_dict = {key: (hf, freeze) for key, hf, freeze in MODEL_SPECS}
    
    if model_key not in spec_dict:
        raise ValueError(f"Unknown model '{model_key}'.")
    
    hf_name, _ = spec_dict[model_key]
    labels = get_labels()
    
    # Load model
    model = TransformerClassifier(hf_name, num_classes=len(labels)).to(DEVICE)
    ckpt_path = CKPT_DIR / f"{model_key}_best.pt"
    
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")
    
    try:
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    except TypeError:
        state = torch.load(ckpt_path, map_location=DEVICE)
    
    model.load_state_dict(state)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    return tokenizer, model


def classify(text: str, model_key: str = DEFAULT_MODEL) -> dict[str, float]:
    """Classify text and return class probabilities."""
    labels = get_labels()
    tokenizer, model = load_model(model_key)
    
    # Preprocess
    text = preprocess(text)
    
    # Tokenize
    inputs = tokenizer(
        text,
        max_length=MAX_LENGTH,
        truncation=True,
        padding=True,
        return_tensors="pt",
    ).to(DEVICE)
    
    # Inference
    with torch.no_grad():
        logits = model(**inputs)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    
    return {label: float(prob) for label, prob in zip(labels, probs)}
