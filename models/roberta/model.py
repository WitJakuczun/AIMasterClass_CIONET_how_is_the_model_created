
import pandas as pd
from mlops.model_interface import ModelInterface
from mlops.pipeline import Pipeline, PipelineContext
from .pipeline import RobertaDataPreprocessor, RobertaTrainer, RobertaPredictor

class RobertaModel(ModelInterface):
    """
    A wrapper for the RoBERTa model that implements the ModelInterface using a pipeline.
    """

    def train(self, train_dataset: pd.DataFrame, hyperparameters: dict, output_dir: str, val_dataset: pd.DataFrame = None):
        """
        Trains a RoBERTa model using a pipeline.
        """
        training_pipeline = Pipeline([
            RobertaDataPreprocessor(),
            RobertaTrainer(),
        ])
        
        context = PipelineContext()
        context.set("train_dataset", train_dataset)
        context.set("hyperparameters", hyperparameters)
        context.set("output_dir", output_dir)
        context.set("val_dataset", val_dataset)
        
        training_pipeline.run(context)

    def predict(self, model_dir: str, data_to_predict: pd.DataFrame, output_file: str):
        """
        Generates predictions using a trained RoBERTa model via a pipeline.
        """
        prediction_pipeline = Pipeline([
            RobertaPredictor(),
        ])
        
        context = PipelineContext()
        context.set("model_dir", model_dir)
        context.set("data_to_predict", data_to_predict)
        context.set("output_file", output_file)
        
        prediction_pipeline.run(context)
