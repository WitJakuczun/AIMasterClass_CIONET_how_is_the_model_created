
from loguru import logger
import pandas as pd
from mlops.model_interface import ModelInterface

class GemmaGGUFModel(ModelInterface):
    """
    A wrapper for the GGUF model that implements the ModelInterface.
    Training is not supported, and prediction is not yet implemented.
    """

    def train(self, train_dataset: pd.DataFrame, hyperparameters: dict, output_dir: str):
        logger.error("Training is not supported for GGUF models.")
        logger.error("GGUF models are pre-quantized and used for inference only.")

    def predict(self, model_dir: str, data_to_predict: pd.DataFrame, output_dir: str):
        raise NotImplementedError("Prediction is not yet implemented for GGUF models.")
