# This script will generate CV splits.

import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
import typer
from loguru import logger

# --- Typer App ---
app = typer.Typer()

@app.command()
def generate_cv_splits(n_splits: int = typer.Option(5, help="Number of splits for cross-validation."),
                     experiment_id: str = typer.Option(..., help="Unique ID for the experiment."),
                     input_file: str = typer.Option("data/all_train.csv", help="Path to the input CSV file."),
                     target_column: str = typer.Option("Sentiment", help="Column to stratify on."),
                     test_size: float = typer.Option(0.2, help="Proportion of the dataset to include in the test split.")):
    """
    Generate Stratified K-Fold CV splits with a hold-out test set.
    """
    # --- Configuration ---
    BASE_OUTPUT_DIR = "experiments"

    # --- Directory Setup ---
    EXPERIMENT_DIR = os.path.join(BASE_OUTPUT_DIR, experiment_id)

    if os.path.exists(EXPERIMENT_DIR):
        logger.error(f"Experiment directory '{EXPERIMENT_DIR}' already exists. Please choose a different experiment ID.")
        raise typer.Exit(code=1)

    os.makedirs(EXPERIMENT_DIR)
    logger.info(f"Created experiment directory: {EXPERIMENT_DIR}")

    # --- Data Loading ---
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        logger.error(f"Input file not found at '{input_file}'")
        raise typer.Exit(code=1)

    if target_column not in df.columns:
        logger.error(f"Target column '{target_column}' not found in the dataframe.")
        raise typer.Exit(code=1)

    # --- Train/Test Split ---
    train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df[target_column])
    test_output_path = os.path.join(EXPERIMENT_DIR, "test.csv")
    test_df.to_csv(test_output_path, index=False)
    logger.info(f"Saved test set ({len(test_df)} rows) to {test_output_path}")

    # --- Stratified Splitting ---
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    logger.info(f"\nGenerating {n_splits} stratified splits from the training data based on '{target_column}' column...")

    for fold, (train_index, val_index) in enumerate(skf.split(train_val_df, train_val_df[target_column])):
        train_df = train_val_df.iloc[train_index]
        val_df = train_val_df.iloc[val_index]

        # --- File Saving ---
        train_output_path = os.path.join(EXPERIMENT_DIR, f"train_fold_{fold}.csv")
        val_output_path = os.path.join(EXPERIMENT_DIR, f"val_fold_{fold}.csv")

        train_df.to_csv(train_output_path, index=False)
        val_df.to_csv(val_output_path, index=False)

        logger.info(f"  - Fold {fold}: Saved train ({len(train_df)} rows) and validation ({len(val_df)} rows) sets.")

    logger.info(f"\nSuccessfully generated and saved {n_splits} CV splits and a test set in '{EXPERIMENT_DIR}'.")

@app.command()
def generate_split(experiment_id: str = typer.Option(..., help="Unique ID for the experiment."),
                   input_file: str = typer.Option("data/all_train.csv", help="Path to the input CSV file."),
                   target_column: str = typer.Option("Sentiment", help="Column to stratify on."),
                   train_ratio: float = typer.Option(0.7, help="Proportion of the dataset for training."),
                   val_ratio: float = typer.Option(0.15, help="Proportion of the dataset for validation.")):
    """
    Generate a single train/validation/test split.
    """
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio < 0 or test_ratio > 1:
        logger.error(f"Invalid split ratios. Train: {train_ratio}, Validation: {val_ratio}, Test: {test_ratio}. Ratios must sum to 1.")
        raise typer.Exit(code=1)

    # --- Configuration ---
    BASE_OUTPUT_DIR = "experiments"

    # --- Directory Setup ---
    EXPERIMENT_DIR = os.path.join(BASE_OUTPUT_DIR, experiment_id)

    if os.path.exists(EXPERIMENT_DIR):
        logger.error(f"Experiment directory '{EXPERIMENT_DIR}' already exists. Please choose a different experiment ID.")
        raise typer.Exit(code=1)

    os.makedirs(EXPERIMENT_DIR)
    logger.info(f"Created experiment directory: {EXPERIMENT_DIR}")

    # --- Data Loading ---
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        logger.error(f"Input file not found at '{input_file}'")
        raise typer.Exit(code=1)

    if target_column not in df.columns:
        logger.error(f"Target column '{target_column}' not found in the dataframe.")
        raise typer.Exit(code=1)

    # --- Train/Validation/Test Split ---
    train_df, temp_df = train_test_split(df, test_size=(val_ratio + test_ratio), random_state=42, stratify=df[target_column])
    val_df, test_df = train_test_split(temp_df, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42, stratify=temp_df[target_column])

    # --- File Saving ---
    train_output_path = os.path.join(EXPERIMENT_DIR, "train.csv")
    val_output_path = os.path.join(EXPERIMENT_DIR, "val.csv")
    test_output_path = os.path.join(EXPERIMENT_DIR, "test.csv")

    train_df.to_csv(train_output_path, index=False)
    val_df.to_csv(val_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)

    logger.info(f"Saved train set ({len(train_df)} rows) to {train_output_path}")
    logger.info(f"Saved validation set ({len(val_df)} rows) to {val_output_path}")
    logger.info(f"Saved test set ({len(test_df)} rows) to {test_output_path}")
    logger.info(f"\nSuccessfully generated and saved train/validation/test splits in '{EXPERIMENT_DIR}'.")

if __name__ == "__main__":
    app()
