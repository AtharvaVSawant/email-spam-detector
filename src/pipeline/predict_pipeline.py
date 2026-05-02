import os
import sys
import pickle
from dataclasses import dataclass
from tensorflow.keras.models import load_model
from src.components.data_transformation import DataTransformation
from src.logger import logging
from src.exception import CustomException


@dataclass
class PredictPipelineConfig:
    model_path: str = os.path.join("artifacts", "lstm_model.h5")
    tokenizer_path: str = os.path.join("artifacts", "tokenizer.pkl")


class PredictPipeline:
    def __init__(self):
        self.config = PredictPipelineConfig()
        self.model = None
        self.tokenizer = None
        self.transformer = DataTransformation()

    def load_model(self):
        if not os.path.exists(self.config.model_path):
            raise FileNotFoundError("Model not found. Train first.")
        self.model = load_model(self.config.model_path)

    def load_tokenizer(self):
        if not os.path.exists(self.config.tokenizer_path):
            raise FileNotFoundError("Tokenizer not found. Train first.")
        with open(self.config.tokenizer_path, "rb") as f:
            self.tokenizer = pickle.load(f)
        self.transformer.tokenizer = self.tokenizer

    def predict(self, text: str):
        try:
            if self.model is None:
                self.load_model()

            if self.tokenizer is None:
                self.load_tokenizer()

            padded = self.transformer.transform_single(text)

            prob = float(self.model.predict(padded)[0][0])
            is_spam = prob >= 0.5

            return {
                "label": "spam" if is_spam else "ham",
                "spam_probability": prob,
                "ham_probability": 1 - prob,
            }

        except Exception as e:
            raise CustomException(e, sys)