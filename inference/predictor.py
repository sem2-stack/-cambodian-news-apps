"""Inference pipeline for the Streamlit dashboard.

This mirrors training exactly (IMPLEMENTATION_GUIDE Part 3.3):

    raw text
      -> preprocessing.clean.preprocess   (lowercase -> strip noise -> stop words)
      -> AutoTokenizer at max_length=512  (body-only, same as training)
      -> TransformerClassifier forward    (LogSoftmax head)
      -> .exp() -> probabilities over the 5 classes

The four ``full_no_environment`` checkpoints are plain ``state_dict`` files, so
we rebuild the architecture with the same ``(hf_name, freeze_until)`` spec used
during training before loading weights.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import torch
from transformers import AutoTokenizer

# app/ holds the dashboard code and the bundled model checkpoints.
APP_DIR = Path(__file__).resolve().parents[1]
# Project root is still needed to reuse the exact training/preprocessing modules.
ROOT = APP_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing.clean import preprocess  # noqa: E402
from training.classifier import TransformerClassifier  # noqa: E402
from training.runner import MODEL_SPECS, load_label_meta  # noqa: E402

CORPUS_ID = "undersampling_no_environment"
SPLITS_DIR = ROOT / "data" / "splits" / CORPUS_ID
CKPT_DIR = APP_DIR / "models" / CORPUS_ID
MAX_LENGTH = 512

# CPU keeps the demo robust across machines; DistilBERT stays well under the
# 5 s latency target and the other encoders are still comfortable.
DEVICE = torch.device("cpu")

# key -> (hf_name, freeze_until)
_SPEC: dict[str, tuple[str, int | None]] = {
    key: (hf, freeze) for key, hf, freeze in MODEL_SPECS
}

# Human-friendly names + test-set metrics for the undersampling_no_environment corpus.
MODEL_INFO: dict[str, dict] = {
    "roberta": {"display": "RoBERTa", "accuracy": 0.9175, "macro_f1": 0.9177},
    "distilbert": {"display": "DistilBERT", "accuracy": 0.9107, "macro_f1": 0.9111},
    "electra": {"display": "ELECTRA", "accuracy": 0.8746, "macro_f1": 0.8761},
    "bert": {"display": "BERT", "accuracy": 0.8418, "macro_f1": 0.8429},
}

DEFAULT_MODEL = "distilbert"


@lru_cache(maxsize=1)
def get_labels() -> list[str]:
    """Ordered class names straight from ``label2id.json`` (index == class id)."""
    labels, _ = load_label_meta(SPLITS_DIR)
    return labels


def available_models() -> list[str]:
    """Model keys whose checkpoint file is present on disk."""
    return [key for key in MODEL_INFO if (CKPT_DIR / f"{key}_best.pt").is_file()]


@lru_cache(maxsize=4)
def load_model(model_key: str):
    """Build the architecture and load the fine-tuned weights (cached once)."""
    if model_key not in _SPEC:
        raise ValueError(f"Unknown model '{model_key}'.")
    hf_name, freeze_until = _SPEC[model_key]
    labels = get_labels()

    model = TransformerClassifier(
        hf_name, freeze_until, num_classes=len(labels)
    ).to(DEVICE)
    ckpt = CKPT_DIR / f"{model_key}_best.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    try:
        state = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    except TypeError:  # older torch without weights_only kwarg
        state = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    return tokenizer, model


def classify(text: str, model_key: str = DEFAULT_MODEL) -> dict[str, float]:
    """Return ``{class_name: probability}`` for one article body."""
    labels = get_labels()
    tokenizer, model = load_model(model_key)

    clean = preprocess(text)
    enc = tokenizer(
        clean,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        log_probs = model(input_ids, attention_mask)
    # Head already ends in LogSoftmax -> exp() gives a proper probability vector.
    probs = log_probs.exp().squeeze(0).tolist()
    return dict(zip(labels, probs))
