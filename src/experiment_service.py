
import os
import pandas as pd
import glob
from typing import List, Tuple
from loguru import logger
from src.experiment import CVExperiment

class ExperimentService:
    """
    Service for handling experiment-related operations like loading and saving.
    """

    def load_experiment(self, experiment_id: str, experiment_path: str) -> CVExperiment:
        """
        Loads the train/validation fold data from the experiment directory.
        """
        experiment = CVExperiment(experiment_id=experiment_id, path=experiment_path)
        
        if self._is_cv_experiment(experiment_path):
            train_files = sorted(glob.glob(os.path.join(experiment_path, "train_fold_*.csv")))
            val_files = sorted(glob.glob(os.path.join(experiment_path, "val_fold_*.csv")))

            if not train_files or not val_files or len(train_files) != len(val_files):
                logger.warning(f"Could not find or match fold files in {experiment_path}")
                return experiment

            for train_file, val_file in zip(train_files, val_files):
                train_df = pd.read_csv(train_file)
                val_df = pd.read_csv(val_file)
                experiment.folds.append((train_df, val_df))
        else:
            train_file = os.path.join(experiment_path, "train.csv")
            val_file = os.path.join(experiment_path, "val.csv")
            if os.path.exists(train_file) and os.path.exists(val_file):
                train_df = pd.read_csv(train_file)
                val_df = pd.read_csv(val_file)
                experiment.folds.append((train_df, val_df))
            else:
                logger.warning(f"Could not find train.csv and val.csv in {experiment_path}")
        
        return experiment

    def _is_cv_experiment(self, path: str) -> bool:
        """Checks if the experiment is a cross-validation experiment."""
        train_files = glob.glob(os.path.join(path, "train_fold_*.csv"))
        return len(train_files) > 0
