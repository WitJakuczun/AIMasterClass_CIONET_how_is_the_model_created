from pathlib import Path
import glob
import pandas as pd
from loguru import logger

from mlops.splitting import DataSplitter
from evaluate import compare_models
from mlops.result_store import ResultStore
from mlops.runner import ExperimentRunner

class Application:
    def __init__(self, paths_config, models_config):
        self.paths = paths_config
        self.models = models_config
        self.runner = ExperimentRunner(paths_config, models_config)
        self.splitter = DataSplitter()

    def generate_splits(self, experiment_id: str, input_file: str, target_column: str, test_size: float, backtesting_strategy: str, cv_folds: int, backtesting_val_size: float, backtesting_test_size: float, perf_estimation_val_size: float, final_model_val_size: float):
        """
        Generates a complete set of data splits for an experiment.
        """
        base_dir = Path(self.paths.experiments) / experiment_id
        backtesting_dir = base_dir / "backtesting"
        perf_estimation_dir = base_dir / "performance_estimation"
        final_model_dir = base_dir / "final_model"

        backtesting_dir.mkdir(parents=True, exist_ok=True)
        perf_estimation_dir.mkdir(parents=True, exist_ok=True)
        final_model_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Created directory structure for experiment '{experiment_id}'")

        df = pd.read_csv(input_file)

        train_val_df, test_df = self.splitter.split_train_test(df, test_size, target_column)
        if test_df is not None:
            test_csv_path = base_dir / "test.csv"
            test_df.to_csv(test_csv_path, index=False)
            logger.info(f"Saved test set to {test_csv_path}")

        logger.info(f"Generating backtesting splits with strategy: '{backtesting_strategy}'...")
        if backtesting_strategy == 'cv':
            cv_dir = backtesting_dir / "cv"
            cv_dir.mkdir(exist_ok=True)
            for fold, train_fold_df, val_fold_df, test_fold_df in self.splitter.split_cv(train_val_df, cv_folds, target_column):
                train_fold_df.to_csv(cv_dir / f"train_fold_{fold}.csv", index=False)
                val_fold_df.to_csv(cv_dir / f"val_fold_{fold}.csv", index=False)
                test_fold_df.to_csv(cv_dir / f"test_fold_{fold}.csv", index=False)
            logger.info(f"- Generated {cv_folds} CV splits in {cv_dir}")
        elif backtesting_strategy == 'train-val':
            train_val_dir = backtesting_dir / "train-val"
            train_val_dir.mkdir(exist_ok=True)
            train_df, val_df, test_df = self.splitter.split_train_val(train_val_df, backtesting_val_size, backtesting_test_size, target_column)
            train_df.to_csv(train_val_dir / "train.csv", index=False)
            val_df.to_csv(train_val_dir / "val.csv", index=False)
            test_df.to_csv(train_val_dir / "test.csv", index=False)
            logger.info(f"- Generated train/val/test split in {train_val_dir}")

        logger.info("Generating performance estimation splits...")
        # This split does not need a separate test set, as evaluation is done on the global hold-out set
        train_df, val_df = self.splitter.split_train_test(train_val_df, perf_estimation_val_size, target_column)
        train_df.to_csv(perf_estimation_dir / "train.csv", index=False)
        val_df.to_csv(perf_estimation_dir / "val.csv", index=False)
        logger.info(f"- Generated train/val split in {perf_estimation_dir}")

        logger.info("Generating final model splits...")
        # This split also uses the entire dataset (minus a small validation set), no test set needed
        train_df, val_df = self.splitter.split_train_test(df, final_model_val_size, target_column)
        train_df.to_csv(final_model_dir / "train.csv", index=False)
        val_df.to_csv(final_model_dir / "val.csv", index=False)
        logger.info(f"- Generated train/val split in {final_model_dir}")

    def run_backtesting(self, experiment_id: str, model_config_name: str):
        self.runner.run_backtesting(experiment_id, model_config_name)

    def estimate_performance(self, experiment_id: str, model_config_name: str):
        self.runner.run_performance_estimation(experiment_id, model_config_name)

    def train_final_model(self, experiment_id: str, model_config_name: str, model_output_dir: str):
        self.runner.train_final_model(experiment_id, model_config_name, model_output_dir)

    def predict_new(self, model_path: str, input_file: str, output_file: str):
        self.runner.predict_new(model_path, input_file, output_file)

    def compare_models(self, experiment_id: str, output_file: str = None, run_type: str = None):
        compare_models(experiment_id, output_file, run_type)

    def list_experiments(self):
        logger.info("Available experiments:")
        experiment_paths = glob.glob(str(Path(self.paths.experiments) / "exp*"))
        
        if not experiment_paths:
            logger.info("No experiments found.")
            return

        for exp_path_str in experiment_paths:
            exp_path = Path(exp_path_str)
            experiment_id = exp_path.name
            print(f"\n--- Experiment: {experiment_id} ---")

            backtesting_path = exp_path / "backtesting"
            cv_path = backtesting_path / "cv"
            train_val_path = backtesting_path / "train-val"
            
            exp_type = "Unknown"
            folds = "N/A"
            if cv_path.exists():
                exp_type = "Cross-Validation"
                fold_files = glob.glob(str(cv_path / "train_fold_*.csv"))
                folds = len(fold_files)
            elif train_val_path.exists():
                exp_type = "Train/Validation"
                folds = 1

            print(f"  Type: {exp_type}")
            print(f"  Folds: {folds}")

            backtesting_results_store = ResultStore(str(backtesting_path))
            backtesting_results = backtesting_results_store.load_results()
            
            if backtesting_results:
                print("  Backtesting Runs:")
                for fold, models in backtesting_results.items():
                    model_names = list(models.keys())
                    print(f"    - Fold {fold}: Models run - {', '.join(model_names)}")

            perf_estimation_path = exp_path / "performance_estimation"
            perf_results_store = ResultStore(str(perf_estimation_path))
            perf_results = perf_results_store.load_results()

            if perf_results:
                print("  Performance Estimation Runs:")
                for fold, models in perf_results.items():
                    model_names = list(models.keys())
                    print(f"    - Models run: {', '.join(model_names)}")

    def list_models(self):
        logger.info("Available models:")
        for name in self.models.keys():
            print(f"- {name}")