import typer
from models_config import MODELS
from loguru import logger
from config import paths
from src.app import Application
from src import runner

typer_app = typer.Typer()

# Instantiate the application
app = Application(runner_module=runner)

@typer_app.command(name="list-models")
def list_models_command():
    """Lists all available models."""
    logger.info("Available models:")
    for model_name in MODELS.keys():
        logger.info(f"- {model_name}")

@typer_app.command(name="generate-splits")
def generate_splits_command(experiment_id: str = typer.Option(..., help="Unique ID for the experiment."),
                            input_file: str = typer.Option(str(paths.all_train_csv), help="Path to the input CSV file."),
                            target_column: str = typer.Option("Sentiment", help="Column to stratify on."),
                            test_size: float = typer.Option(0.2, help="Proportion of the dataset for the hold-out test set."),
                            backtesting_strategy: str = typer.Option("cv", help="Backtesting strategy: 'cv' or 'train-val'."),
                            cv_folds: int = typer.Option(5, help="Number of folds for cross-validation."),
                            backtesting_val_size: float = typer.Option(0.15, help="Validation set size for the backtesting train/val split."),
                            perf_estimation_val_size: float = typer.Option(0.1, help="Validation set size for the performance estimation split."),
                            final_model_val_size: float = typer.Option(0.1, help="Validation set size for the final model training split.")):
    app.generate_splits(experiment_id, input_file, target_column, test_size, backtesting_strategy, cv_folds, backtesting_val_size, perf_estimation_val_size, final_model_val_size)

@typer_app.command(name="run-backtesting")
def run_backtesting_command(experiment_id: str = typer.Option(..., help="Unique ID for the experiment."),
                            model_config_name: str = typer.Option(..., help="Name of the model config to use.")):
    app.run_backtesting(experiment_id, model_config_name)

@typer_app.command(name="estimate-performance")
def estimate_performance_command(experiment_id: str = typer.Option(..., help="Unique ID for the experiment."),
                                 model_config_name: str = typer.Option(..., help="Name of the model config to use.")):
    app.estimate_performance(experiment_id, model_config_name)

@typer_app.command(name="train-final-model")
def train_final_model_command(experiment_id: str = typer.Option(..., help="Unique ID for the experiment."),
                              model_config_name: str = typer.Option(..., help="Name of the model config to use."),
                              model_output_dir: str = typer.Option(str(paths.trained_models), help="Directory to save the final model.")):
    app.train_final_model(experiment_id, model_config_name, model_output_dir)

@typer_app.command(name="train")
def train_command(experiment_id: str = typer.Option(..., help="Unique ID for the experiment."),
                  fold_number: int = typer.Option(..., help="Fold number to train on."),
                  model_config_name: str = typer.Option("gemma-3-1b-it", help="Name of the model config to use from models_config.py"),                  
                  model_output_dir: str = typer.Option(str(paths.trained_models), help="Directory to save the trained model.")):
    """
    Train a model on a specific fold of an experiment.
    """
    app.train_model(experiment_id, fold_number, model_config_name, model_output_dir)


@typer_app.command(name="predict")
def predict_command(experiment_id: str = typer.Option(..., help="Unique ID for the experiment."),
                    fold_number: int = typer.Option(..., help="Fold number to predict on."),
                    model_config_name: str = typer.Option("gemma-3-1b-it", help="Name of the model config to use from models_config.py"),
                    model_input_dir: str = typer.Option(str(paths.trained_models), help="Directory where the trained model is saved."),
                    prediction_output_dir: str = typer.Option(str(paths.predictions), help="Directory to save the predictions.")):
    """
    Predict on a specific fold of an experiment.
    """
    app.predict_model(experiment_id, fold_number, model_config_name, model_input_dir, prediction_output_dir)


@typer_app.command(name="evaluate")
def evaluate_command(experiment_id: str = typer.Option(..., help="Unique ID for the experiment."),
                     fold_number: int = typer.Option(..., help="Fold number to evaluate on."),
                     prediction_input_dir: str = typer.Option(str(paths.predictions), help="Directory where the predictions are saved.")) -> dict:
    """
    Evaluate the predictions for a specific fold of an experiment.
    """
    return app.evaluate_predictions(experiment_id, fold_number, prediction_input_dir)

@typer_app.command(name="run-experiment")
def run_experiment_command(experiment_id: str = typer.Option(..., help="Unique ID for the experiment."),
                           model_config_name: str = typer.Option("gemma-2b", help="Name of the model to use from models_config.py"),
                           model_output_dir: str = typer.Option(str(paths.trained_models), help="Directory to save the trained model."),
                           prediction_output_dir: str = typer.Option(str(paths.predictions), help="Directory to save the predictions.")):
    """
    Run a full experiment: train, predict, and evaluate for all folds.
    """
    app.run_experiment(experiment_id, model_config_name, model_output_dir, prediction_output_dir)

@typer_app.command(name="predict-new")
def predict_new_command(model_path: str = typer.Option(..., help="Path to the trained model."),
                        input_file: str = typer.Option(..., help="Path to the input CSV file for predictions."),
                        output_file: str = typer.Option(..., help="Path to save the predictions.")):
    """
    Make predictions using a trained model on new data.
    """
    app.predict_new_data(model_path, input_file, output_file)

@typer_app.command(name="compare-models")
def compare_models_command(experiment_id: str = typer.Option(..., help="Unique ID for the experiment."),
                           output_file: str = typer.Option(None, help="Path to save the comparison report."),
                           run_type: str = typer.Option(None, help="Run type to compare: 'backtesting' or 'performance_estimation'.")):
    """
    Compare the performance of different models for a given experiment.
    """
    app.compare_models(experiment_id, output_file, run_type)

if __name__ == "__main__":
    typer_app()