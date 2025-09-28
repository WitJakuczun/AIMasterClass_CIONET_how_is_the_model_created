
import typer
from models_config import MODELS
from loguru import logger
from config import paths
from src import runner
from src.app import Application

typer_app = typer.Typer()

# Instantiate the application
app = Application(runner_module=runner)

@typer_app.command(name="list-models")
def list_models_command():
    """Lists all available models."""
    logger.info("Available models:")
    for model_name in MODELS.keys():
        logger.info(f"- {model_name}")

@typer_app.command(name="generate-cv-splits")
def generate_cv_splits_command(n_splits: int = typer.Option(5, help="Number of splits for cross-validation."),
                                experiment_id: str = typer.Option(..., help="Unique ID for the experiment."),
                                input_file: str = typer.Option(str(paths.all_train_csv), help="Path to the input CSV file."),
                                target_column: str = typer.Option("Sentiment", help="Column to stratify on."),
                                test_size: float = typer.Option(0.2, help="Proportion of the dataset to include in the test split.")):
    app.generate_cv_splits(n_splits, experiment_id, input_file, target_column, test_size)

@typer_app.command(name="generate-split")
def generate_split_command(experiment_id: str = typer.Option(..., help="Unique ID for the experiment."),
                           input_file: str = typer.Option(str(paths.all_train_csv), help="Path to the input CSV file."),
                           target_column: str = typer.Option("Sentiment", help="Column to stratify on."),
                           train_ratio: float = typer.Option(0.7, help="Proportion of the dataset for training."),
                           val_ratio: float = typer.Option(0.15, help="Proportion of the dataset for validation.")):
    app.generate_split(experiment_id, input_file, target_column, train_ratio, val_ratio)

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

if __name__ == "__main__":
    typer_app()
