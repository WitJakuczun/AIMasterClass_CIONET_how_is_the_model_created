
import pandas as pd
from sklearn.metrics import accuracy_score

def evaluate(predictions_file: str, ground_truth_file: str) -> dict:
    """
    Evaluates the predictions against the ground truth.

    Args:
        predictions_file: Path to the predictions CSV file.
        ground_truth_file: Path to the ground truth CSV file.
    
    Returns:
        A dictionary containing the evaluation metrics.
    """
    
    predictions_df = pd.read_csv(predictions_file)
    ground_truth_df = pd.read_csv(ground_truth_file)
    
    accuracy = accuracy_score(ground_truth_df["Sentiment"], predictions_df["Predicted_Sentiment"])
    
    return {"accuracy": accuracy}
