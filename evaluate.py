import polars as pl
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from mlops.result_store import ResultStore
import os
import glob
import json
from loguru import logger

def evaluate(predictions_file: str, ground_truth_file: str) -> dict:
    """
    Evaluates predictions and returns a dictionary of metrics.
    """
    # Load predictions and ground truth
    predictions_df = pl.read_csv(predictions_file)
    ground_truth_df = pl.read_csv(ground_truth_file)

    # Polars doesn't have index merge, we assume they are aligned by row if no common key
    # Or horizontal concat if they have same height
    if predictions_df.height != ground_truth_df.height:
        logger.warning(f"Predictions ({predictions_df.height}) and ground truth ({ground_truth_df.height}) have different heights!")
    
    # We'll just take the columns directly.
    y_true = ground_truth_df['Sentiment']
    y_pred = predictions_df['Predicted_Sentiment']

    # Calculate metrics
    accuracy = accuracy_score(y_true.to_numpy(), y_pred.to_numpy())
    precision, recall, f1, _ = precision_recall_fscore_support(y_true.to_numpy(), y_pred.to_numpy(), average='weighted')

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    return metrics

def compare_models(experiment_id: str, output_file: str = None, run_type: str = None):
    """
    Compare the performance of different models for a given experiment.
    Returns a list of dictionaries containing the results.
    """
    from config import paths
    experiment_path = os.path.join(paths.experiments, experiment_id)
    
    run_types_to_process = []
    if run_type:
        run_types_to_process.append(run_type)
    else:
        run_types_to_process.extend(["backtesting", "performance_estimation"])

    all_results = []
    for rt in run_types_to_process:
        results_file = os.path.join(experiment_path, rt, "results.json")
        if not os.path.exists(results_file):
            logger.warning(f"No results file found for run type '{rt}' in experiment '{experiment_id}'.")
            continue

        with open(results_file, 'r') as f:
            data = json.load(f)

        for fold, models in data.items():
            for model_name, metrics in models.items():
                row = {"run_type": rt, "model_name": model_name, "fold": fold, **metrics}
                all_results.append(row)

    if not all_results:
        logger.warning(f"No results found for experiment '{experiment_id}'.")
        return []

    results_df = pl.DataFrame(all_results)

    # Convert metrics to numeric, ensuring types
    metrics_cols = ['accuracy', 'precision', 'recall', 'f1_score']
    for col in metrics_cols:
        if col in results_df.columns:
            results_df = results_df.with_columns(pl.col(col).cast(pl.Float64))

    # Display results
    logger.info(f"--- Model Comparison for Experiment '{experiment_id}' ---")
    
    run_types = results_df["run_type"].unique().to_list()
    for rt in run_types:
        logger.info(f"\n--- Run Type: {rt} ---")
        group_df = results_df.filter(pl.col("run_type") == rt)
        
        if rt == "backtesting":
            if group_df['fold'].n_unique() > 1: # CV strategy
                # Polars doesn't have pivot_table exactly like pandas, but has pivot
                per_fold_df = group_df.pivot(index=['model_name', 'fold'], on=[], values=['accuracy', 'precision', 'recall', 'f1_score'])
                logger.info(f"Per-fold results for {rt}:\n{per_fold_df}")
                
                logger.info("\nSummary (mean and std across folds):")
                summary_df = group_df.group_by('model_name').agg([
                    pl.col(c).mean().alias(f"{c}_mean") for c in metrics_cols
                ] + [
                    pl.col(c).std().alias(f"{c}_std") for c in metrics_cols
                ])
            else: # Train-val strategy
                logger.info(f"Summary for {rt} (train-val strategy):")
                summary_df = group_df.group_by('model_name').agg([
                    pl.col(c).mean().alias(f"{c}_mean") for c in metrics_cols
                ])
        else:
            summary_df = group_df.group_by('model_name').agg([
                pl.col(c).mean().alias(f"{c}_mean") for c in metrics_cols
            ] + [
                pl.col(c).std().alias(f"{c}_std") for c in metrics_cols
            ])
        logger.info("\n" + str(summary_df))

    if output_file:
        results_df = results_df.sort("accuracy", descending=True)
        results_df.write_csv(output_file)
        logger.info(f"Comparison report saved to {output_file}")
    
    return all_results
