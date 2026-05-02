from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainingPipeline:

    def run(self):
        train_path, test_path = DataIngestion().initiate_data_ingestion()

        X_train, X_test, y_train, y_test = DataTransformation().fit_transform(
            train_path, test_path
        )

        ModelTrainer().train(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    TrainingPipeline().run()