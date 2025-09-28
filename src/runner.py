
import os
import numpy as np
from loguru import logger
from models_config import MODELS
from config import paths
import importlib.util
import inspect
from src.model_interface import ModelInterface
from evaluate import evaluate as evaluate_model
from src.experiment_service import ExperimentService
from src.result_store import ResultStore

def load_model_from_config(model_config) -> ModelInterface:
    """
    Dynamically loads a model class from a given model config.
    """
    model_path = model_config.model_path
    module_path = os.path.join(model_path, "model.py")
    
    # Construct a proper module name for handling relative imports
    # Assuming module_path is like /path/to/project/models/gemma/model.py
    # We want a module name like models.gemma.model
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    relative_path = os.path.relpath(module_path, project_root)
    # Remove .py extension and replace / with .
    module_full_name = relative_path.replace(os.sep, '.')[:-3] 
    
    spec = importlib.util.spec_from_file_location(module_full_name, module_path)
    if spec is None:
        raise ImportError(f"Could not load spec for module at path {module_path}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and issubclass(obj, ModelInterface) and obj is not ModelInterface:
            return obj()
            
    raise TypeError(f"No class implementing ModelInterface found in {module_path}")

def train_model(experiment_id: str, fold_number: int, model_config_name: str, model_output_dir: str):
    """
    Train a model on a specific fold of an experiment.
    """
    experiment_service = ExperimentService()
    experiment_path = os.path.join(paths.experiments, experiment_id)
    experiment = experiment_service.load_experiment(experiment_id, experiment_path)
    
    train_df, _ = experiment.get_fold(fold_number)
    
    model_config = MODELS[model_config_name]
    
    output_dir = os.path.join(model_output_dir, f"{experiment_id}_fold_{fold_number}")

    try:
        model = load_model_from_config(model_config)
        model.train(train_dataset=train_df, hyperparameters=model_config.__dict__, output_dir=output_dir)
    except (ImportError, TypeError, Exception) as e:
        logger.error(f"Error loading or running train module: {e}")

def predict_model(experiment_id: str, fold_number: int, model_config_name: str, model_input_dir: str, prediction_output_dir: str):
    """
    Predict on a specific fold of an experiment.
    """
    experiment_service = ExperimentService()
    experiment_path = os.path.join(paths.experiments, experiment_id)
    experiment = experiment_service.load_experiment(experiment_id, experiment_path)
    
    _, val_df = experiment.get_fold(fold_number)
    
    model_dir = os.path.join(model_input_dir, f"{experiment_id}_fold_{fold_number}")
    output_dir = os.path.join(prediction_output_dir, f"{experiment_id}_fold_{fold_number}")
    
    model_config = MODELS[model_config_name]

    try:
        model = load_model_from_config(model_config)
        model.predict(model_dir=model_dir, data_to_predict=val_df, output_dir=output_dir)
    except (ImportError, TypeError, Exception) as e:
        logger.error(f"Error loading or running predict module: {e}")


def evaluate_predictions(experiment_id: str, fold_number: int, model_config_name: str, prediction_input_dir: str) -> dict:
    """
    Evaluate the predictions for a specific fold of an experiment.
    """
    experiment_service = ExperimentService()
    experiment_path = os.path.join(paths.experiments, experiment_id)
    experiment = experiment_service.load_experiment(experiment_id, experiment_path)

    prediction_file = os.path.join(prediction_input_dir, f"{experiment_id}_fold_{fold_number}", "predictions.csv")
    
    if experiment.is_cv_experiment:
        ground_truth_file = os.path.join(experiment_path, f"val_fold_{fold_number}.csv")
    else:
        ground_truth_file = os.path.join(experiment_path, "val.csv")
    
    metrics = evaluate_model(predictions_file=prediction_file, ground_truth_file=ground_truth_file)
    logger.info(f"Fold {fold_number} metrics: {metrics}")

    result_store = ResultStore(experiment_path)
    result_store.save_metrics(fold=fold_number, model_name=model_config_name, metrics=metrics)

    return metrics

def run_experiment(experiment_id: str, model_config_name: str, model_output_dir: str, prediction_output_dir: str):
    """
    Run a full experiment: train, predict, and evaluate for all folds.
    """
    experiment_service = ExperimentService()
    experiment_path = os.path.join(paths.experiments, experiment_id)
    experiment = experiment_service.load_experiment(experiment_id, experiment_path)
    
    all_metrics = []
    for i in range(experiment.n_splits):
        logger.info(f"--- Running Fold {i} ---")
        train_model(experiment_id, i, model_config_name, model_output_dir)
        predict_model(experiment_id, i, model_config_name, model_output_dir, prediction_output_dir)
        metrics = evaluate_predictions(experiment_id, i, model_config_name, prediction_output_dir)
        all_metrics.append(metrics)

    logger.info("--- Experiment Summary ---")
    
    if experiment.is_cv_experiment:
        # --- Display results in a table ---
        headers = ["Fold"] + list(all_metrics[0].keys())
        
        # Header
        header_line = " | ".join(f"{h:<15}" for h in headers)
        logger.info(header_line)
        logger.info("-" * len(header_line))

        # Rows
        for i, metrics in enumerate(all_metrics):
            row_values = [f"Fold {i}"] + list(metrics.values())
            row_line = " | ".join(f"{v:<15.4f}" if isinstance(v, float) else f"{str(v):<15}" for v in row_values)
            logger.info(row_line)

        # --- Summary ---
        logger.info("-" * len(header_line))
        summary_values = ["Summary"] 
        for key in headers[1:]:
            values = [m[key] for m in all_metrics]
            mean = np.mean(values)
            std = np.std(values)
            summary_values.append(f"{mean:.4f} +/- {std:.4f}")
        
        summary_line = " | ".join(f"{v:<15}" for v in summary_values)
        logger.info(summary_line)
    else:
        logger.info(f"Metrics: {all_metrics[0]}")
