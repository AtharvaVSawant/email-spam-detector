import os
import pandas as pd
import urllib.request
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
import sys


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self, data_path: str = None):
        """
        Load SMS Spam Collection dataset.
        If data_path is None, downloads from UCI repository.
        """
        logging.info("Starting data ingestion")
        try:
            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)

            if data_path and os.path.exists(data_path):
                df = pd.read_csv(data_path, encoding="latin-1")
            else:
                # Use bundled sample or download
                logging.info("Loading SMS Spam Collection dataset")
                url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv"
                df = pd.read_csv(url, sep="\t", header=None, names=["label", "message"])

            # Standardise columns
            if "v1" in df.columns and "v2" in df.columns:
                df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "message"})
            elif "label" not in df.columns:
                df.columns = ["label", "message"] + list(df.columns[2:])

            df = df[["label", "message"]].dropna()
            df["label"] = df["label"].map({"ham": 0, "spam": 1})

            df.to_csv(self.config.raw_data_path, index=False)
            logging.info(f"Raw data saved: {self.config.raw_data_path} — {len(df)} rows")

            # Train/test split
            from sklearn.model_selection import train_test_split
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
            train_df.to_csv(self.config.train_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)
            logging.info("Train/test split complete")

            return self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)
