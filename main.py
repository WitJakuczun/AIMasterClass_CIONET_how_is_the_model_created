import typer
from typing_extensions import Annotated
from loguru import logger

# Configuration imports
from config import paths
from models_config import MODELS

# Module imports
from mlops.app import Application
from data.generator import DataGenerator

app = typer.Typer()

# Initialize the application with configuration
application = Application(paths_config=paths, models_config=MODELS)

@app.command()
def data_generate(
    name: Annotated[str, typer.Option(help="Name of the output file (without extension)")],
    examples_per_sentiment: Annotated[int, typer.Option(help="Number of examples to generate for each sentiment")] = None,
    distribution_from_file: Annotated[str, typer.Option(help="Path to a CSV file to read sentiment distribution from")] = None,
    total_examples: Annotated[int, typer.Option(help="Total number of new examples to generate with the specified distribution")] = None,
    lang: Annotated[str, typer.Option(help="Language for the generated text (e.g., 'pl', 'de')")] = "en"
):
    """
    Generates new data for sentiment analysis using the Gemma model.
    """
    # Argument validation
    if examples_per_sentiment and distribution_from_file:
        logger.error("Error: --examples-per-sentiment and --distribution-from-file are mutually exclusive.")
        raise typer.Exit(code=1)
    
    if distribution_from_file and not total_examples:
        logger.error("Error: --total-examples is required when using --distribution-from-file.")
        raise typer.Exit(code=1)

    if not examples_per_sentiment and not distribution_from_file:
        logger.error("Error: You must specify a generation mode: either --examples-per-sentiment or --distribution-from-file.")
        raise typer.Exit(code=1)

    try:
        generator = DataGenerator()
        generator.generate(
            name=name,
            examples_per_sentiment=examples_per_sentiment,
            distribution_from_file=distribution_from_file,
            total_examples=total_examples,
            lang=lang
        )
    except Exception as e:
        logger.error(f"An error occurred during data generation: {e}")
        raise typer.Exit(code=1)

@app.command()
def generate_splits(
    experiment_id: Annotated[str, typer.Option(help="Unique ID for the experiment")],
    input_file: Annotated[str, typer.Option(help="Path to the input CSV file")],
    target_column: Annotated[str, typer.Option(help="Column to stratify on")] = 'Sentiment',
    test_size: Annotated[float, typer.Option(help="Proportion for the initial hold-out test set")] = 0.2,
    backtesting_strategy: Annotated[str, typer.Option(help="Backtesting strategy: 'cv' or 'train-val'")] = 'cv',
    cv_folds: Annotated[int, typer.Option(help="Number of folds for cross-validation")] = 5,
    backtesting_val_size: Annotated[float, typer.Option(help="Validation set size for backtesting train/val split")] = 0.15,
    perf_estimation_val_size: Annotated[float, typer.Option(help="Validation set size for performance estimation split")] = 0.1,
    final_model_val_size: Annotated[float, typer.Option(help="Validation set size for final model training split")] = 0.1
):
    """
    Generates a complete set of data splits for an experiment.
    """
    application.generate_splits(
        experiment_id=experiment_id,
        input_file=input_file,
        target_column=target_column,
        test_size=test_size,
        backtesting_strategy=backtesting_strategy,
        cv_folds=cv_folds,
        backtesting_val_size=backtesting_val_size,
        perf_estimation_val_size=perf_estimation_val_size,
        final_model_val_size=final_model_val_size
    )

@app.command()
def run_backtesting(
    experiment_id: Annotated[str, typer.Option(help="Unique ID for the experiment")],
    model_config_name: Annotated[str, typer.Option(help="Name of the model configuration to use")]
):
    """
    Runs the backtesting phase for a given experiment and model.
    """
    application.run_backtesting(experiment_id, model_config_name)

@app.command()
def estimate_performance(
    experiment_id: Annotated[str, typer.Option(help="Unique ID for the experiment")],
    model_config_name: Annotated[str, typer.Option(help="Name of the model configuration to use")]
):
    """
    Runs the performance estimation phase for a given experiment and model.
    """
    application.estimate_performance(experiment_id, model_config_name)

@app.command()
def train_final_model(
    experiment_id: Annotated[str, typer.Option(help="Unique ID for the experiment")],
    model_config_name: Annotated[str, typer.Option(help="Name of the model configuration to use")],
    model_output_dir: Annotated[str, typer.Option(help="Directory to save the trained model")]
):
    """
    Trains a final model on the full training data.
    """
    application.train_final_model(experiment_id, model_config_name, model_output_dir)

@app.command()
def predict_new(
    model_path: Annotated[str, typer.Option(help="Path to the trained model directory")],
    input_file: Annotated[str, typer.Option(help="Path to the CSV file with data to predict")],
    output_file: Annotated[str, typer.Option(help="Path to save the predictions CSV file")]
):
    """
    Makes predictions on new data using a trained model.
    """
    application.predict_new(model_path, input_file, output_file)

@app.command()
def compare_models(
    experiment_id: Annotated[str, typer.Option(help="Unique ID for the experiment")],
    output_file: Annotated[str, typer.Option(help="Path to save the comparison report")] = None,
    run_type: Annotated[str, typer.Option(help="Run type to compare: 'backtesting' or 'performance_estimation'")] = None
):
    """
    Compares the performance of different models for a given experiment.
    """
    application.compare_models(experiment_id, output_file, run_type)

@app.command()
def list_models():
    """
    Lists all available models.
    """
    application.list_models()

@app.command()
def list_experiments():
    """
    Lists all available experiments with their statistics.
    """
    application.list_experiments()

if __name__ == "__main__":
    app()
