

import os
import pandas as pd
from loguru import logger
from config import paths
from models_config import MODELS
from evaluate import evaluate
import importlib
import glob

from mlops.model_interface import ModelInterface
from mlops.experiment_service import ExperimentService
from mlops.result_store import ResultStore

def load_model_from_config(model_config) -> ModelInterface:
    """
    Loads a model dynamically from a given module path and class name.
    """
    try:
        module_path = model_config.module_path
        class_name = model_config.class_name
        
        logger.info(f"Loading model: {class_name} from {module_path}")
        
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
        
        return model_class()
    except (ImportError, AttributeError) as e:
        logger.error(f"Error loading model: {e}")
        raise

def run_backtesting(experiment_id: str, model_config_name: str):
    """
    Runs the backtesting phase for a given experiment and model.
    """
    exp_service = ExperimentService(experiment_id)
    model_config = MODELS[model_config_name]
    result_store = ResultStore(exp_service.get_backtesting_path())

    for fold, (train_df, val_df) in enumerate(exp_service.get_backtesting_folds()):
        logger.info(f"--- Backtesting Fold {fold} ---")
        
        model_output_dir = os.path.join(paths.trained_models, f"{experiment_id}_backtesting", f"fold_{fold}", model_config_name)
        prediction_output_dir = os.path.join(paths.predictions, f"{experiment_id}_backtesting", f"fold_{fold}", model_config_name)
        os.makedirs(model_output_dir, exist_ok=True)
        os.makedirs(prediction_output_dir, exist_ok=True)

        try:
            model = load_model_from_config(model_config)
            
            # Train
            logger.info("Training model...")
            model.train(train_dataset=train_df, val_dataset=val_df, hyperparameters=model_config.__dict__, output_dir=model_output_dir)
            
            # Predict
            logger.info("Predicting...")
            model.predict(model_dir=model_output_dir, data_to_predict=val_df, output_dir=prediction_output_dir)
            
            # Evaluate
            logger.info("Evaluating...")
            prediction_file = os.path.join(prediction_output_dir, "predictions.csv")
            ground_truth_file = exp_service.get_backtesting_fold_path(fold)['val']
            metrics = evaluate(predictions_file=prediction_file, ground_truth_file=ground_truth_file)
            
            logger.info(f"Metrics for fold {fold}: {metrics}")
            result_store.save_metrics(fold=fold, model_name=model_config_name, metrics=metrics)

        except Exception as e:
            logger.error(f"Error during backtesting fold {fold}: {e}")

def run_performance_estimation(experiment_id: str, model_config_name: str):
    """
    Runs the performance estimation phase for a given experiment and model.
    """
    exp_service = ExperimentService(experiment_id)
    model_config = MODELS[model_config_name]
    result_store = ResultStore(exp_service.get_performance_estimation_path())

    train_df, val_df = exp_service.get_performance_estimation_fold()
    
    model_output_dir = os.path.join(paths.trained_models, f"{experiment_id}_performance_estimation", model_config_name)
    prediction_output_dir = os.path.join(paths.predictions, f"{experiment_id}_performance_estimation", model_config_name)
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(prediction_output_dir, exist_ok=True)

    try:
        model = load_model_from_config(model_config)
        
        # Train
        logger.info("Training model for performance estimation...")
        model.train(train_dataset=train_df, val_dataset=val_df, hyperparameters=model_config.__dict__, output_dir=model_output_dir)
        
        # Predict on test set
        logger.info("Predicting on test set...")
        test_df = exp_service.get_test_set()
        model.predict(model_dir=model_output_dir, data_to_predict=test_df, output_dir=prediction_output_dir)
        
        # Evaluate
        logger.info("Evaluating performance on test set...")
        prediction_file = os.path.join(prediction_output_dir, "predictions.csv")
        ground_truth_file = exp_service.get_test_set_path()
        metrics = evaluate(predictions_file=prediction_file, ground_truth_file=ground_truth_file)
        
        logger.info(f"Performance estimation metrics: {metrics}")
        result_store.save_metrics(fold=0, model_name=model_config_name, metrics=metrics)

    except Exception as e:
        logger.error(f"Error during performance estimation: {e}")


def train_final_model(experiment_id: str, model_config_name: str, model_output_dir: str):
    """
    Trains a final model on the full training data.
    """
    exp_service = ExperimentService(experiment_id)
    model_config = MODELS[model_config_name]

    train_df, val_df = exp_service.get_final_model_fold()
    
    final_model_output_dir = os.path.join(model_output_dir, f"{experiment_id}_final", model_config_name)
    os.makedirs(final_model_output_dir, exist_ok=True)

    try:
        model = load_model_from_config(model_config)
        
        logger.info("Training final model...")
        model.train(train_dataset=train_df, val_dataset=val_df, hyperparameters=model_config.__dict__, output_dir=final_model_output_dir)
        logger.success(f"Final model trained and saved to {final_model_output_dir}")

    except Exception as e:
        logger.error(f"Error during final model training: {e}")

def predict_new(model_path: str, input_file: str, output_file: str):
    """
    Makes predictions on new data using a trained model.
    """
    # This function needs to know which model to load. This information should be stored with the model.
    # For now, we assume a RoBERTa model, but this needs to be generalized.
    # A potential solution is to save model_config.json next to the model.
    
    # Dynamically load the model class from the model path
    # This is a placeholder - a more robust mechanism is needed
    from models.roberta.model import RobertaModel as ModelClass
    
    model = ModelClass()
    data_to_predict = pd.read_csv(input_file)
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    model.predict(model_dir=model_path, data_to_predict=data_to_predict, output_dir=output_dir)
    
    # Rename the generic predictions.csv to the desired output file name
    os.rename(os.path.join(output_dir, "predictions.csv"), output_file)
    logger.success(f"Predictions saved to {output_file}")
