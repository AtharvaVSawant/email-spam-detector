import os
import sys
import pickle
import numpy as np
from dataclasses import dataclass
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.components.data_transformation import DataTransformation
from src.logger import logging
from src.exception import CustomException


@dataclass
class PredictPipelineConfig:
    model_path: str = os.path.join("artifacts", "lstm_model.h5")
    tokenizer_path: str = os.path.join("artifacts", "tokenizer.pkl")
    max_len: int = 150


class PredictPipeline:
    """Loads trained LSTM model + tokenizer and serves single-text predictions."""

    def __init__(self):
        self.config = PredictPipelineConfig()
        self._model = None
        self._tokenizer = None
        self._transformer = DataTransformation()

    @property
    def model(self):
        if self._model is None:
            if not os.path.exists(self.config.model_path):
                raise FileNotFoundError(
                    f"Model not found at {self.config.model_path}. "
                    "Run the training pipeline first."
                )
            logging.info("Loading LSTM model …")
            self._model = load_model(self.config.model_path)
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            if not os.path.exists(self.config.tokenizer_path):
                raise FileNotFoundError(
                    f"Tokenizer not found at {self.config.tokenizer_path}. "
                    "Run the training pipeline first."
                )
            logging.info("Loading tokenizer …")
            with open(self.config.tokenizer_path, "rb") as f:
                self._tokenizer = pickle.load(f)
            self._transformer.tokenizer = self._tokenizer
        return self._tokenizer

    def predict(self, text: str) -> dict:
        """
        Predict whether a message is spam.

        Returns:
            {
                "label": "spam" | "ham",
                "is_spam": bool,
                "spam_probability": float,   # 0-1
                "confidence": float,          # 0-1 (distance from 0.5)
            }
        """
        try:
            _ = self.tokenizer  # ensure tokenizer loaded into transformer
            padded = self._transformer.transform_single(text)
            prob = float(self.model.predict(padded, verbose=0)[0][0])
            is_spam = prob >= 0.5
            confidence = abs(prob - 0.5) * 2  # re-scale to 0-1

            return {
                "label": "spam" if is_spam else "ham",
                "is_spam": is_spam,
                "spam_probability": prob,
                "ham_probability": 1 - prob,
                "confidence": confidence,
            }
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """Helper to package raw user input for the prediction pipeline."""

    def __init__(self, message: str):
        self.message = message

    def as_dict(self):
        return {"message": self.message}
