import os
import glob
import pandas as pd
from loguru import logger
from src import runner
from src.splitting import DataSplitter
from evaluate import compare_models, evaluate
from config import paths
from models_config import MODELS
from src.result_store import ResultStore

class Application:
    def __init__(self, runner_module):
        self.runner = runner_module
        self.splitter = DataSplitter()

    def generate_splits(self, experiment_id: str, input_file: str, target_column: str, test_size: float, backtesting_strategy: str, cv_folds: int, backtesting_val_size: float, perf_estimation_val_size: float, final_model_val_size: float):
        """
        Generates a complete set of data splits for an experiment.
        """
        # Create experiment directory structure
        base_dir = os.path.join(paths.experiments, experiment_id)
        backtesting_dir = os.path.join(base_dir, "backtesting")
        perf_estimation_dir = os.path.join(base_dir, "performance_estimation")
        final_model_dir = os.path.join(base_dir, "final_model")

        os.makedirs(backtesting_dir, exist_ok=True)
        os.makedirs(perf_estimation_dir, exist_ok=True)
        os.makedirs(final_model_dir, exist_ok=True)

        logger.info(f"Created directory structure for experiment '{experiment_id}'")

        # Load data
        df = pd.read_csv(input_file)

        # Initial train/test split
        train_val_df, test_df = self.splitter.split_train_test(df, test_size, target_column)
        if test_df is not None:
            test_df.to_csv(os.path.join(base_dir, "test.csv"), index=False)
            logger.info(f"Saved test set to {os.path.join(base_dir, 'test.csv')}")

        # Backtesting splits
        logger.info(f"Generating backtesting splits with strategy: '{backtesting_strategy}'...")
        if backtesting_strategy == 'cv':
            cv_dir = os.path.join(backtesting_dir, "cv")
            os.makedirs(cv_dir, exist_ok=True)
            for fold, train_fold_df, val_fold_df in self.splitter.split_cv(train_val_df, cv_folds, target_column):
                train_fold_df.to_csv(os.path.join(cv_dir, f"train_fold_{fold}.csv"), index=False)
                val_fold_df.to_csv(os.path.join(cv_dir, f"val_fold_{fold}.csv"), index=False)
            logger.info(f"- Generated {cv_folds} CV splits in {cv_dir}")
        elif backtesting_strategy == 'train-val':
            train_val_dir = os.path.join(backtesting_dir, "train-val")
            os.makedirs(train_val_dir, exist_ok=True)
            train_df, val_df = self.splitter.split_train_val(train_val_df, backtesting_val_size, target_column)
            train_df.to_csv(os.path.join(train_val_dir, "train.csv"), index=False)
            val_df.to_csv(os.path.join(train_val_dir, "val.csv"), index=False)
            logger.info(f"- Generated train/val split in {train_val_dir}")

        # Performance estimation splits
        logger.info("Generating performance estimation splits...")
        train_df, val_df = self.splitter.split_train_val(train_val_df, perf_estimation_val_size, target_column)
        train_df.to_csv(os.path.join(perf_estimation_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(perf_estimation_dir, "val.csv"), index=False)
        logger.info(f"- Generated train/val split in {perf_estimation_dir}")

        # Final model splits
        logger.info("Generating final model splits...")
        train_df, val_df = self.splitter.split_train_val(df, final_model_val_size, target_column)
        train_df.to_csv(os.path.join(final_model_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(final_model_dir, "val.csv"), index=False)
        logger.info(f"- Generated train/val split in {final_model_dir}")

    def run_backtesting(self, experiment_id: str, model_config_name: str):
        logger.info(f"Running backtesting for experiment '{experiment_id}' with model '{model_config_name}'")
        
        experiment_dir = os.path.join(paths.experiments, experiment_id)
        backtesting_dir = os.path.join(experiment_dir, "backtesting")
        cv_dir = os.path.join(backtesting_dir, "cv")
        train_val_dir = os.path.join(backtesting_dir, "train-val")

        if os.path.exists(cv_dir):
            logger.info("CV strategy detected.")
            fold_files = glob.glob(os.path.join(cv_dir, "train_fold_*.csv"))
            for fold, train_file in enumerate(fold_files):
                val_file = os.path.join(cv_dir, f"val_fold_{fold}.csv")
                self._run_training_and_evaluation(
                    train_file=train_file,
                    val_file=val_file,
                    model_config_name=model_config_name,
                    experiment_id=experiment_id,
                    fold=fold
                )
        elif os.path.exists(train_val_dir):
            logger.info("Train-val strategy detected.")
            train_file = os.path.join(train_val_dir, "train.csv")
            val_file = os.path.join(train_val_dir, "val.csv")
            self._run_training_and_evaluation(
                train_file=train_file,
                val_file=val_file,
                model_config_name=model_config_name,
                experiment_id=experiment_id
            )
        else:
            logger.error(f"No backtesting splits found for experiment '{experiment_id}'. Please run generate-splits first.")

    def estimate_performance(self, experiment_id: str, model_config_name: str):
        logger.info(f"Estimating performance for experiment '{experiment_id}' with model '{model_config_name}'")
        
        experiment_dir = os.path.join(paths.experiments, experiment_id)
        perf_estimation_dir = os.path.join(experiment_dir, "performance_estimation")
        train_file = os.path.join(perf_estimation_dir, "train.csv")
        test_file = os.path.join(experiment_dir, "test.csv")

        self._run_training_and_evaluation(
            train_file=train_file,
            val_file=test_file,
            model_config_name=model_config_name,
            experiment_id=experiment_id,
            run_type="performance_estimation"
        )

    def train_final_model(self, experiment_id: str, model_config_name: str, model_output_dir: str):
        logger.info(f"Training final model for experiment '{experiment_id}' with model '{model_config_name}'")

        experiment_dir = os.path.join(paths.experiments, experiment_id)
        final_model_dir = os.path.join(experiment_dir, "final_model")
        train_file = os.path.join(final_model_dir, "train.csv")
        val_file = os.path.join(final_model_dir, "val.csv")
        
        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file) if os.path.exists(val_file) else None

        model_config = MODELS[model_config_name]
        output_dir = os.path.join(model_output_dir, f"{experiment_id}_final", model_config_name)

        try:
            model = self.runner.load_model_from_config(model_config)
            model.train(train_dataset=train_df, val_dataset=val_df, hyperparameters=model_config.__dict__, output_dir=output_dir)
            logger.success(f"Final model trained and saved to {output_dir}")
        except (ImportError, TypeError, Exception) as e:
            logger.error(f"Error loading or running train module: {e}")

    def _run_training_and_evaluation(self, train_file: str, val_file: str, model_config_name: str, experiment_id: str, fold: int = None, run_type: str = "backtesting"):
        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file)

        model_config = MODELS[model_config_name]
        
        model_output_dir = os.path.join(paths.trained_models, f"{experiment_id}_{run_type}", model_config_name)
        prediction_output_dir = os.path.join(paths.predictions, f"{experiment_id}_{run_type}", model_config_name)
        
        if fold is not None:
            model_output_dir = os.path.join(model_output_dir, f"fold_{fold}")
            prediction_output_dir = os.path.join(prediction_output_dir, f"fold_{fold}")

        try:
            model = self.runner.load_model_from_config(model_config)
            
            # Train
            model.train(train_dataset=train_df, val_dataset=val_df, hyperparameters=model_config.__dict__, output_dir=model_output_dir)
            
            # Predict
            model.predict(model_dir=model_output_dir, data_to_predict=val_df, output_dir=prediction_output_dir)
            
            # Evaluate
            prediction_file = os.path.join(prediction_output_dir, "predictions.csv")
            metrics = evaluate(predictions_file=prediction_file, ground_truth_file=val_file)
            
            logger.info(f"Metrics for {run_type} {'fold ' + str(fold) if fold is not None else ''}: {metrics}")

            # Save results
            result_store = ResultStore(os.path.join(paths.experiments, experiment_id, run_type))
            result_store.save_metrics(fold=fold if fold is not None else 0, model_name=model_config_name, metrics=metrics)

        except (ImportError, TypeError, Exception) as e:
            logger.error(f"Error during training and evaluation: {e}")

    def run_experiment(self, experiment_id: str, model_config_name: str, model_output_dir: str, prediction_output_dir: str):
        self.runner.run_experiment(experiment_id, model_config_name, model_output_dir, prediction_output_dir)

    def train_model(self, experiment_id: str, fold_number: int, model_config_name: str, model_output_dir: str):
        self.runner.train_model(experiment_id, fold_number, model_config_name, model_output_dir)

    def predict_model(self, experiment_id: str, fold_number: int, model_config_name: str, model_input_dir: str, prediction_output_dir: str):
        self.runner.predict_model(experiment_id, fold_number, model_config_name, model_input_dir, prediction_output_dir)

    def evaluate_predictions(self, experiment_id: str, fold_number: int, prediction_input_dir: str):
        return self.runner.evaluate_predictions(experiment_id, fold_number, prediction_input_dir)

    def compare_models(self, experiment_id: str, output_file: str, run_type: str = None):
        compare_models(experiment_id, output_file, run_type)

    def predict_new_data(self, model_path: str, input_file: str, output_file: str):
        self.runner.predict_new_data(model_path, input_file, output_file)