import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException


class DataIngestion:

    def initiate_data_ingestion(self):
        try:
            logging.info("Starting data ingestion")

            data_path = os.path.join("data", "spam.csv")

            if not os.path.exists(data_path):
                raise FileNotFoundError(f"{data_path} not found")

            df = pd.read_csv(data_path, encoding="latin-1")

            df = df.iloc[:, :2]
            df.columns = ["label", "message"]

            os.makedirs("artifacts", exist_ok=True)

            train_df, test_df = train_test_split(
                df,
                test_size=0.2,
                random_state=42,
                stratify=df["label"]
            )

            train_path = os.path.join("artifacts", "train.csv")
            test_path = os.path.join("artifacts", "test.csv")

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            return train_path, test_path

        except Exception as e:
            raise CustomException(e, sys)