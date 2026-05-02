import os
import sys
import json
import numpy as np
from dataclasses import dataclass
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, LSTM, Dense, Dropout, Bidirectional, GlobalMaxPooling1D
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from src.logger import logging
from src.exception import CustomException


@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join("artifacts", "lstm_model.h5")
    metrics_path: str = os.path.join("artifacts", "metrics.json")
    # Hyperparameters
    embedding_dim: int = 128
    lstm_units: int = 64
    dropout_rate: float = 0.3
    learning_rate: float = 1e-3
    epochs: int = 15
    batch_size: int = 64
    max_words: int = 10000
    max_len: int = 150


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def build_model(self) -> Sequential:
        """Bidirectional LSTM with dropout for robust spam classification."""
        model = Sequential([
            Embedding(
                input_dim=self.config.max_words,
                output_dim=self.config.embedding_dim,
                input_length=self.config.max_len,
                mask_zero=True
            ),
            Bidirectional(LSTM(self.config.lstm_units, return_sequences=True)),
            Dropout(self.config.dropout_rate),
            Bidirectional(LSTM(self.config.lstm_units // 2)),
            Dropout(self.config.dropout_rate),
            Dense(32, activation="relu"),
            Dropout(self.config.dropout_rate),
            Dense(1, activation="sigmoid"),
        ])
        model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        model.summary()
        return model

    def train(self, X_train, X_test, y_train, y_test):
        logging.info("Starting model training")
        try:
            os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)

            model = self.build_model()

            callbacks = [
                EarlyStopping(patience=3, restore_best_weights=True, monitor="val_loss"),
                ModelCheckpoint(self.config.model_path, save_best_only=True, monitor="val_loss"),
                ReduceLROnPlateau(patience=2, factor=0.5, monitor="val_loss", verbose=1),
            ]

            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                callbacks=callbacks,
                verbose=1
            )

            # Evaluate
            y_prob = model.predict(X_test).flatten()
            y_pred = (y_prob >= 0.5).astype(int)

            report = classification_report(y_test, y_pred, output_dict=True)
            auc = roc_auc_score(y_test, y_prob)
            cm = confusion_matrix(y_test, y_pred).tolist()

            metrics = {
                "accuracy": report["accuracy"],
                "precision_spam": report["1"]["precision"],
                "recall_spam": report["1"]["recall"],
                "f1_spam": report["1"]["f1-score"],
                "roc_auc": auc,
                "confusion_matrix": cm,
                "history": {
                    "train_acc": history.history["accuracy"],
                    "val_acc": history.history["val_accuracy"],
                    "train_loss": history.history["loss"],
                    "val_loss": history.history["val_loss"],
                }
            }

            with open(self.config.metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)

            logging.info(f"Model saved → {self.config.model_path}")
            logging.info(f"Accuracy: {metrics['accuracy']:.4f} | AUC: {auc:.4f}")

            return model, metrics

        except Exception as e:
            raise CustomException(e, sys)
