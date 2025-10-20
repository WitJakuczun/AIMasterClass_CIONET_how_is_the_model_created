

from pathlib import Path
import pandas as pd
from loguru import logger
from evaluate import evaluate
import importlib
import json
from types import SimpleNamespace

from mlops.model_interface import ModelInterface
from mlops.experiment_service import ExperimentService, RunType
from mlops.result_store import ResultStore

class ExperimentRunner:
    def __init__(self, paths_config: object, models_config: object):
        self.paths = paths_config
        self.models = models_config

    def _load_model_from_config(self, model_config) -> ModelInterface:
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

    def _save_model_spec(self, model_config, output_dir: Path):
        """Saves the model's spec (module and class name) to a JSON file."""
        spec = {
            "module_path": model_config.module_path,
            "class_name": model_config.class_name
        }
        spec_path = output_dir / "model_spec.json"
        with spec_path.open('w') as f:
            json.dump(spec, f, indent=4)
        logger.info(f"Model spec saved to {spec_path}")

    def run_backtesting(self, experiment_id: str, model_config_name: str):
        """
        Runs the backtesting phase for a given experiment and model.
        """
        exp_service = ExperimentService(experiment_id, self.paths)
        model_config = self.models[model_config_name]
        result_store = ResultStore(exp_service.get_run_path(RunType.BACKTESTING))

        for fold, (train_df, val_df, test_df) in enumerate(exp_service.get_backtesting_folds()):
            logger.info(f"--- Backtesting Fold {fold} ---")
            
            model_output_dir = Path(self.paths.trained_models) / f"{experiment_id}_backtesting" / f"fold_{fold}" / model_config_name
            prediction_output_dir = Path(self.paths.predictions) / f"{experiment_id}_backtesting" / f"fold_{fold}" / model_config_name
            model_output_dir.mkdir(parents=True, exist_ok=True)
            prediction_output_dir.mkdir(parents=True, exist_ok=True)

            try:
                model = self._load_model_from_config(model_config)
                
                logger.info("Training model...")
                model.train(train_dataset=train_df, val_dataset=val_df, hyperparameters=model_config.__dict__, output_dir=str(model_output_dir))
                self._save_model_spec(model_config, model_output_dir)
                
                logger.info("Predicting...")
                model.predict(model_dir=str(model_output_dir), data_to_predict=test_df, output_dir=str(prediction_output_dir))
                
                logger.info("Evaluating...")
                prediction_file = prediction_output_dir / "predictions.csv"
                ground_truth_file = exp_service.get_backtesting_fold_path(fold)['test']
                
                metrics = evaluate(predictions_file=str(prediction_file), ground_truth_file=ground_truth_file)
                
                logger.info(f"Metrics for fold {fold}: {metrics}")
                result_store.save_metrics(fold=fold, model_name=model_config_name, metrics=metrics)

            except Exception as e:
                logger.error(f"Error during backtesting fold {fold}: {e}")

    def run_performance_estimation(self, experiment_id: str, model_config_name: str):
        """
        Runs the performance estimation phase for a given experiment and model.
        """
        exp_service = ExperimentService(experiment_id, self.paths)
        model_config = self.models[model_config_name]
        result_store = ResultStore(exp_service.get_run_path(RunType.PERFORMANCE_ESTIMATION))

        train_df, val_df = exp_service.get_train_val_split(RunType.PERFORMANCE_ESTIMATION)
        
        model_output_dir = Path(self.paths.trained_models) / f"{experiment_id}_performance_estimation" / model_config_name
        prediction_output_dir = Path(self.paths.predictions) / f"{experiment_id}_performance_estimation" / model_config_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        prediction_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            model = self._load_model_from_config(model_config)
            
            logger.info("Training model for performance estimation...")
            model.train(train_dataset=train_df, val_dataset=val_df, hyperparameters=model_config.__dict__, output_dir=str(model_output_dir))
            self._save_model_spec(model_config, model_output_dir)
            
            logger.info("Predicting on test set...")
            test_df = exp_service.get_test_set()
            model.predict(model_dir=str(model_output_dir), data_to_predict=test_df, output_dir=str(prediction_output_dir))
            
            logger.info("Evaluating performance on test set...")
            prediction_file = prediction_output_dir / "predictions.csv"
            ground_truth_file = exp_service.get_test_set_path()
            metrics = evaluate(predictions_file=str(prediction_file), ground_truth_file=ground_truth_file)
            
            logger.info(f"Performance estimation metrics: {metrics}")
            result_store.save_metrics(fold=0, model_name=model_config_name, metrics=metrics)

        except Exception as e:
            logger.error(f"Error during performance estimation: {e}")

    def train_final_model(self, experiment_id: str, model_config_name: str, model_output_dir: str):
        """
        Trains a final model on the full training data.
        """
        exp_service = ExperimentService(experiment_id, self.paths)
        model_config = self.models[model_config_name]

        train_df, val_df = exp_service.get_train_val_split(RunType.FINAL_MODEL)
        
        final_model_output_dir = Path(model_output_dir) / f"{experiment_id}_final" / model_config_name
        final_model_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            model = self._load_model_from_config(model_config)
            
            logger.info("Training final model...")
            model.train(train_dataset=train_df, val_dataset=val_df, hyperparameters=model_config.__dict__, output_dir=str(final_model_output_dir))
            self._save_model_spec(model_config, final_model_output_dir)
            logger.success(f"Final model trained and saved to {final_model_output_dir}")

        except Exception as e:
            logger.error(f"Error during final model training: {e}")

    def predict_new(self, model_path: str, input_file: str, output_file: str):
        """
        Makes predictions on new data using a trained model.
        """
        model_path = Path(model_path)
        spec_path = model_path / "model_spec.json"

        if not spec_path.exists():
            logger.error(f"Model spec file not found at {spec_path}")
            raise FileNotFoundError(f"Model spec file not found at {spec_path}")

        with spec_path.open('r') as f:
            spec_data = json.load(f)
        
        model_config = SimpleNamespace(**spec_data)
        
        model = self._load_model_from_config(model_config)
        data_to_predict = pd.read_csv(input_file)
        output_path = Path(output_file)
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model.predict(model_dir=str(model_path), data_to_predict=data_to_predict, output_dir=str(output_dir))
        
        (output_dir / "predictions.csv").rename(output_path)
        logger.success(f"Predictions saved to {output_file}")
