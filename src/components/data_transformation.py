import os
import re
import sys
import pickle
import pandas as pd
from dataclasses import dataclass
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.logger import logging
from src.exception import CustomException


@dataclass
class DataTransformationConfig:
    tokenizer_path: str = os.path.join("artifacts", "tokenizer.pkl")
    max_words: int = 10000
    max_len: int = 150


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()
        self.tokenizer = Tokenizer(
            num_words=self.config.max_words,
            oov_token="<OOV>"
        )

    def clean_text(self, text: str) -> str:
        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+", " url ", text)
        text = re.sub(r"\d+", " num ", text)
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def fit_transform(self, train_path: str, test_path: str):

        try:
            logging.info("Starting data transformation")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # 🔥 CRITICAL FIX: convert labels to numeric
            train_df["label"] = train_df["label"].map({"ham": 0, "spam": 1})
            test_df["label"] = test_df["label"].map({"ham": 0, "spam": 1})

            # optional safety check
            if train_df["label"].isnull().sum() > 0:
                raise ValueError("Label mapping failed in training data")

            if test_df["label"].isnull().sum() > 0:
                raise ValueError("Label mapping failed in test data")

            # text cleaning
            train_df["clean"] = train_df["message"].apply(self.clean_text)
            test_df["clean"] = test_df["message"].apply(self.clean_text)

            # fit tokenizer
            self.tokenizer.fit_on_texts(train_df["clean"])

            os.makedirs("artifacts", exist_ok=True)

            with open(self.config.tokenizer_path, "wb") as f:
                pickle.dump(self.tokenizer, f)

            logging.info("Tokenizer saved")

            # convert text to sequences
            X_train = pad_sequences(
                self.tokenizer.texts_to_sequences(train_df["clean"]),
                maxlen=self.config.max_len,
                padding="post",
                truncating="post"
            )

            X_test = pad_sequences(
                self.tokenizer.texts_to_sequences(test_df["clean"]),
                maxlen=self.config.max_len,
                padding="post",
                truncating="post"
            )

            # labels (now numeric)
            y_train = train_df["label"].values
            y_test = test_df["label"].values

            logging.info("Data transformation completed")

            return X_train, X_test, y_train, y_test

        except Exception as e:
            raise CustomException(e, sys)

    # ✅ FIX FOR PREDICTION PIPELINE
    def transform_single(self, text: str):
        try:
            clean = self.clean_text(text)

            seq = self.tokenizer.texts_to_sequences([clean])

            padded = pad_sequences(
                seq,
                maxlen=self.config.max_len,
                padding="post",
                truncating="post"
            )

            return padded

        except Exception as e:
            raise CustomException(e, sys)