import os
import json
import pickle
import numpy as np


def save_object(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_object(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_metrics(path: str = os.path.join("artifacts", "metrics.json")) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def artifacts_exist() -> bool:
    """Return True only if both model and tokenizer are present."""
    return (
        os.path.exists(os.path.join("artifacts", "lstm_model.h5"))
        and os.path.exists(os.path.join("artifacts", "tokenizer.pkl"))
    )
