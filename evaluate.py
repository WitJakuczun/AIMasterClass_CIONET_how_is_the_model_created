

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from src.result_store import ResultStore
import os
import glob
import json
from loguru import logger

def evaluate(predictions_file: str, ground_truth_file: str) -> dict:
    """
    Evaluates predictions and returns a dictionary of metrics.
    """
    # Load predictions and ground truth
    predictions_df = pd.read_csv(predictions_file)
    ground_truth_df = pd.read_csv(ground_truth_file)

    # Merge dataframes to align predictions with ground truth based on index
    merged_df = pd.merge(predictions_df, ground_truth_df, left_index=True, right_index=True, suffixes=('_pred', '_true'))

    # Get labels and predictions
    y_true = merged_df['Sentiment_true']
    y_pred = merged_df['Predicted_Sentiment']

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    return metrics

def compare_models(experiment_id: str, output_file: str = None):
    """
    Compare the performance of different models for a given experiment.
    """
    from config import paths
    experiment_path = os.path.join(paths.experiments, experiment_id)
    results_file = os.path.join(experiment_path, "results.json")

    if not os.path.exists(results_file):
        logger.warning(f"No results file found for experiment '{experiment_id}'.")
        return

    with open(results_file, 'r') as f:
        data = json.load(f)

    all_results = []
    for fold, models in data.items():
        for model_name, metrics in models.items():
            row = {"model_name": model_name, "fold": fold, **metrics}
            all_results.append(row)

    results_df = pd.DataFrame(all_results)

    # Convert metrics to numeric, coercing errors
    for col in ['accuracy', 'precision', 'recall', 'f1_score']:
        if col in results_df.columns:
            results_df[col] = pd.to_numeric(results_df[col], errors='coerce')

    # Drop non-numeric columns before aggregation
    summary_df = results_df.drop(columns=['fold']).groupby('model_name').agg(['mean', 'std']).reset_index()

    # Display results
    logger.info(f"--- Model Comparison for Experiment '{experiment_id}' ---")
    logger.info(summary_df.to_string())

    if output_file:
        summary_df.to_csv(output_file, index=False)
        logger.info(f"Comparison report saved to {output_file}")

