from abc import ABC, abstractmethod
import polars as pl


class ModelInterface(ABC):
    """
    Abstract base class for a machine learning model.
    It defines the interface for training and prediction.
    """

    @abstractmethod
    def train(
        self,
        train_dataset: pl.DataFrame,
        hyperparameters: dict,
        output_dir: str,
        val_dataset: pl.DataFrame = None,
    ) -> None:
        """
        Train the model.

        Args:
            train_dataset: The training dataset.
            hyperparameters: A dictionary of hyperparameters.
            output_dir: The directory to save the trained model.
            val_dataset: The validation dataset.
        """
        pass

    @abstractmethod
    def predict(
        self, model_dir: str, data_to_predict: pl.DataFrame, output_file: str
    ) -> None:
        """
        Make predictions using a trained model.

        Args:
            model_dir: The directory where the trained model is saved.
            data_to_predict: The data to make predictions on.
            output_file: The path to the file where the predictions will be saved.
        """
        pass
