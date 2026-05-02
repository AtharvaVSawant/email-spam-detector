import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.exception import CustomException


class TrainPipeline:
    def __init__(self):
        self.ingestion = DataIngestion()
        self.transformation = DataTransformation()
        self.trainer = ModelTrainer()

    def run(self, data_path: str = None):
        """End-to-end training: ingest → transform → train → evaluate."""
        try:
            logging.info("═" * 50)
            logging.info("TRAINING PIPELINE STARTED")
            logging.info("═" * 50)

            # Step 1 — Data Ingestion
            logging.info("Step 1/3: Data Ingestion")
            train_path, test_path = self.ingestion.initiate_data_ingestion(data_path)

            # Step 2 — Data Transformation
            logging.info("Step 2/3: Data Transformation")
            X_train, X_test, y_train, y_test = self.transformation.fit_transform(train_path, test_path)

            # Step 3 — Model Training
            logging.info("Step 3/3: Model Training")
            model, metrics = self.trainer.train(X_train, X_test, y_train, y_test)

            logging.info("═" * 50)
            logging.info("TRAINING PIPELINE COMPLETE")
            logging.info(f"  Accuracy : {metrics['accuracy']:.4f}")
            logging.info(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")
            logging.info(f"  F1 (spam): {metrics['f1_spam']:.4f}")
            logging.info("═" * 50)

            return metrics

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run()
