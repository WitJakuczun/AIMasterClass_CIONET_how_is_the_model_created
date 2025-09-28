
import json
import os
from typing import Dict, Any
from loguru import logger

class ResultStore:
    """
    Handles the storage and retrieval of experiment results.
    """
    def __init__(self, experiment_path: str):
        self.results_file = os.path.join(experiment_path, "results.json")

    def save_metrics(self, fold: int, model_name: str, metrics: Dict[str, Any]):
        """
        Saves the metrics for a given fold and model.
        """
        results = self.load_results()
        
        if str(fold) not in results:
            results[str(fold)] = {}
            
        results[str(fold)][model_name] = metrics
        
        try:
            with open(self.results_file, 'w') as f:
                json.dump(results, f, indent=4)
            logger.info(f"Saved metrics for fold {fold}, model {model_name} to {self.results_file}")
        except IOError as e:
            logger.error(f"Could not write to results file {self.results_file}: {e}")

    def load_results(self) -> Dict[str, Any]:
        """
        Loads the results from the results file.
        """
        if not os.path.exists(self.results_file):
            return {}
        
        try:
            with open(self.results_file, 'r') as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Could not read or parse results file {self.results_file}: {e}")
            return {}
