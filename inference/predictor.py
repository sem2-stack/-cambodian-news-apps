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
    
    # Load checkpoint
    ckpt_path = CKPT_DIR / f"{model_key}_best.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")
    
    try:
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    except TypeError:
        state = torch.load(ckpt_path, map_location=DEVICE)
    
    # If checkpoint is a full model object, extract state_dict
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    elif isinstance(state, nn.Module):
        state = state.state_dict()
    elif not isinstance(state, dict):
        # Try to treat as model and get state_dict
        try:
            state = state.state_dict()
        except:
            pass
    
    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    
    # Create model
    model = TransformerClassifier(hf_name, num_classes=len(labels)).to(DEVICE)
    
    # Load weights with flexibility for architecture mismatches
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        # Try loading with strict=False if there's a mismatch
        try:
            incompatible = model.load_state_dict(state, strict=False)
            if incompatible.missing_keys:
                print(f"Warning: Missing keys in checkpoint: {incompatible.missing_keys}")
            if incompatible.unexpected_keys:
                print(f"Warning: Unexpected keys in checkpoint: {incompatible.unexpected_keys}")
        except Exception as load_err:
            # Last resort: try to load individual components
            print(f"Could not load state dict: {e}")
            print(f"Attempting component-wise loading...")
            
            # Try loading encoder weights
            encoder_state = {k.replace('encoder.', ''): v for k, v in state.items() if k.startswith('encoder.')}
            if encoder_state:
                try:
                    model.encoder.load_state_dict(encoder_state, strict=False)
                    print("Loaded encoder weights successfully")
                except:
                    pass
            
            # Try loading classifier weights
            classifier_state = {k.replace('classifier.', ''): v for k, v in state.items() if k.startswith('classifier.')}
            if classifier_state:
                try:
                    model.classifier.load_state_dict(classifier_state, strict=False)
                    print("Loaded classifier weights successfully")
                except:
                    pass
    
    model.eval()
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
