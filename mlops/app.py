from pathlib import Path
import glob
import polars as pl
from loguru import logger
from rich.console import Console
from rich.table import Table
from mlops.visualization import plot_model_comparison
from mlops.terminal_utils import display_image

from mlops.splitting import DataSplitter
from evaluate import compare_models, evaluate
from mlops.result_store import ResultStore
from mlops.runner import ExperimentRunner
from mlops.experiment_service import ExperimentService, RunType

class Application:
    def __init__(self, paths_config, models_config, console=None):
        self.paths = paths_config
        self.models = models_config
        self.console = console or Console()
        self.runner = ExperimentRunner(paths_config, models_config, self.console)
        self.splitter = DataSplitter()

    def generate_splits(self, experiment_id: str, input_file: str, target_column: str, test_size: float, backtesting_strategy: str, cv_folds: int = 5, backtesting_val_size: float = 0.2, backtesting_test_size: float = 0.2, perf_estimation_val_size: float = 0.2, final_model_val_size: float = 0.05):
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

        exp_service = ExperimentService(experiment_id, self.paths, self.console)
        df = exp_service._load_csv_with_status(Path(input_file), "Loading input data")

        train_val_df, test_df = self.splitter.split_train_test(df, test_size, target_column)
        if test_df is not None:
            test_csv_path = base_dir / "test.csv"
            test_df.write_csv(test_csv_path)
            logger.info(f"Saved test set to {test_csv_path}")

        logger.info(f"Generating backtesting splits with strategy: '{backtesting_strategy}'...")
        if backtesting_strategy == 'cv':
            cv_dir = backtesting_dir / "cv"
            cv_dir.mkdir(exist_ok=True)
            for fold, train_fold_df, val_fold_df, test_fold_df in self.splitter.split_cv(train_val_df, cv_folds, target_column):
                train_fold_df.write_csv(cv_dir / f"train_fold_{fold}.csv")
                val_fold_df.write_csv(cv_dir / f"val_fold_{fold}.csv")
                test_fold_df.write_csv(cv_dir / f"test_fold_{fold}.csv")
            logger.info(f"- Generated {cv_folds} CV splits in {cv_dir}")
        elif backtesting_strategy == 'train-val':
            train_val_dir = backtesting_dir / "train-val"
            train_val_dir.mkdir(exist_ok=True)
            train_df, val_df, test_df = self.splitter.split_train_val(train_val_df, backtesting_val_size, backtesting_test_size, target_column)
            train_df.write_csv(train_val_dir / "train.csv")
            val_df.write_csv(train_val_dir / "val.csv")
            test_df.write_csv(train_val_dir / "test.csv")
            logger.info(f"- Generated train/val/test split in {train_val_dir}")

        logger.info("Generating performance estimation splits...")
        # This split does not need a separate test set, as evaluation is done on the global hold-out set
        train_df, val_df = self.splitter.split_train_test(train_val_df, perf_estimation_val_size, target_column)
        train_df.write_csv(perf_estimation_dir / "train.csv")
        val_df.write_csv(perf_estimation_dir / "val.csv")
        logger.info(f"- Generated train/val split in {perf_estimation_dir}")

        logger.info("Generating final model splits...")
        # This split also uses the entire dataset (minus a small validation set), no test set needed
        train_df, val_df = self.splitter.split_train_test(df, final_model_val_size, target_column)
        train_df.write_csv(final_model_dir / "train.csv")
        val_df.write_csv(final_model_dir / "val.csv")
        logger.info(f"- Generated train/val split in {final_model_dir}")

    def run_backtesting(self, experiment_id: str, model_config_name: str):
        self.runner.run_backtesting(experiment_id, model_config_name)

    def estimate_performance(self, experiment_id: str, model_config_name: str):
        self.runner.run_performance_estimation(experiment_id, model_config_name)

    def train_final_model(self, experiment_id: str, model_config_name: str, model_output_dir: str):
        self.runner.train_final_model(experiment_id, model_config_name, model_output_dir)

    def predict_new(self, model_path: str, input_file: str, prediction_id: str, timestamp: str, name: str):
        self.runner.predict_new(model_path, input_file, prediction_id, timestamp, name)

    def evaluate_prediction(self, predictions_file: str, ground_truth_file: str, timestamp: str, name: str, model_id: str, display_it: bool):
        logger.info(f"Evaluating predictions from '{predictions_file}' against '{ground_truth_file}'.")
        metrics = evaluate(predictions_file, ground_truth_file)
        logger.info(f"Evaluation metrics: {metrics}")
        
        output_dir = Path(self.paths.metrics) / timestamp / name / model_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "metrics.csv"
        
        metrics_df = pl.DataFrame([metrics])
        metrics_df.write_csv(output_file)        
        logger.success(f"Metrics saved to {output_file}")

        if display_it:
            print(metrics_df)

    def visualize_comparison(self, experiment_id: str, run_type: str = "backtesting", metric: str = "f1_score") -> str:
        """
        Generuje wykres porównujący modele w ramach eksperymentu.
        """
        print(f"Generowanie wizualizacji dla: {experiment_id}, {run_type}, {metric}")
        
        # 1. Ustalenie ścieżek
        exp_dir = Path(self.paths.experiments) / experiment_id
        if not exp_dir.exists():
            return f"Eksperyment {experiment_id} nie istnieje."

        # Try loading from ResultStore (JSON) first
        run_dir = exp_dir / run_type
        result_store = ResultStore(run_dir)
        results_data = result_store.load_results() # Returns Dict[fold, Dict[model, metrics]]

        df = pl.DataFrame()
        
        if results_data:
            # Transform JSON to DataFrame
            rows = []
            for fold, models in results_data.items():
                for model_name, metrics in models.items():
                    row = metrics.copy()
                    row['model_name'] = model_name
                    row['fold'] = fold
                    rows.append(row)
            df = pl.DataFrame(rows)
        
        # Fallback to CSV if JSON empty (legacy support)
        if df.is_empty():
            # Podejście: Szukamy results.csv w katalogu run_type
            results_path = exp_dir / run_type / "results.csv"
            
            if not results_path.exists():
                # Fallback: próba znalezienia w metrics jeśli to backtesting
                possible_paths = list(exp_dir.glob(f"**/*{run_type}*.csv"))
                if possible_paths:
                     results_path = possible_paths[0]
            
            if results_path.exists():
                try:
                    df = pl.read_csv(results_path)
                except Exception as e:
                    return f"Błąd odczytu wyników CSV: {str(e)}"

        if df.is_empty():
            return f"Nie znaleziono wyników dla {experiment_id} / {run_type}"
        
        # Verify metric exists BEFORE aggregation
        if metric not in df.columns:
             return f"Metryka {metric} nie dostępna. Dostępne: {list(df.columns)}"

        # Agregacja jeśli to backtesting (średnia z foldów)
        if "fold" in df.columns and "model_name" in df.columns:
            df = df.group_by("model_name").agg(pl.col(metric).mean())
        
        if df.is_empty():
            return "Brak danych do wizualizacji."

        # 2. Przygotowanie ścieżki wyjściowej
        output_dir = exp_dir / "visualizations"
        filename = f"comparison_{run_type}_{metric}.png"
        output_path = output_dir / filename

        # 3. Generowanie wykresu
        try:
            plot_path = plot_model_comparison(
                results=df,
                metric_name=metric,
                output_path=output_path,
                title=f"Comparison: {experiment_id} ({run_type})"
            )
            display_image(str(output_path))
            return f"Wykres został wygenerowany i zapisany w: {plot_path}"
        except Exception as e:
            return f"Błąd generowania wykresu: {str(e)}"

    def compare_models(self, experiment_id: str, output_file: str = None, run_type: str = None):
        return compare_models(experiment_id, output_file, run_type)

    def list_experiments(self):
        experiment_paths = glob.glob(str(Path(self.paths.experiments) / "exp*"))
        
        if not experiment_paths:
            logger.info("No experiments found.")
            return

        for exp_path_str in experiment_paths:
            exp_path = Path(exp_path_str)
            experiment_id = exp_path.name
            
            backtesting_path = exp_path / "backtesting"
            cv_path = backtesting_path / "cv"
            train_val_path = backtesting_path / "train-val"
            
            exp_type = "Unknown"
            folds = "N/A"
            if cv_path.exists():
                exp_type = "Cross-Validation"
                fold_files = glob.glob(str(cv_path / "train_fold_*.csv"))
                folds = str(len(fold_files))
            elif train_val_path.exists():
                exp_type = "Train/Validation"
                folds = "1"

            table = Table(title=f"Experiment: {experiment_id}")
            table.add_column("Property", style="bold")
            table.add_column("Value")
            
            table.add_row("Type", exp_type)
            table.add_row("Folds", folds)

            backtesting_results_store = ResultStore(str(backtesting_path))
            backtesting_results = backtesting_results_store.load_results()
            
            if backtesting_results:
                bt_runs = []
                for fold, models in backtesting_results.items():
                    model_names = list(models.keys())
                    bt_runs.append(f"Fold {fold}: {', '.join(model_names)}")
                table.add_row("Backtesting Runs", "\n".join(bt_runs))

            perf_estimation_path = exp_path / "performance_estimation"
            perf_results_store = ResultStore(str(perf_estimation_path))
            perf_results = perf_results_store.load_results()

            if perf_results:
                pe_runs = []
                for fold, models in perf_results.items():
                    model_names = list(models.keys())
                    pe_runs.append(f"{', '.join(model_names)}")
                table.add_row("Perf Estimation Runs", "\n".join(pe_runs))

            self.console.print(table)
            self.console.print("")

    def list_models(self):
        table = Table(title="Available Models")
        table.add_column("Model Name", style="cyan", no_wrap=True)
        table.add_column("Class", style="magenta")
        table.add_column("Module", style="green")

        for name, config in self.models.items():
            table.add_row(name, config.class_name, config.module_path)
        
        self.console.print(table)
