

import os
import pandas as pd
from loguru import logger
from config import paths
import glob

from mlops.experiment import CVExperiment

class ExperimentService:
    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        self.experiment_path = os.path.join(paths.experiments, self.experiment_id)

    def get_backtesting_path(self) -> str:
        return os.path.join(self.experiment_path, "backtesting")

    def get_performance_estimation_path(self) -> str:
        return os.path.join(self.experiment_path, "performance_estimation")

    def get_final_model_path(self) -> str:
        return os.path.join(self.experiment_path, "final_model")

    def get_backtesting_folds(self):
        backtesting_path = self.get_backtesting_path()
        cv_path = os.path.join(backtesting_path, "cv")
        train_val_path = os.path.join(backtesting_path, "train-val")

        if os.path.exists(cv_path):
            fold_files = glob.glob(os.path.join(cv_path, "train_fold_*.csv"))
            for i in range(len(fold_files)):
                train_df = pd.read_csv(os.path.join(cv_path, f"train_fold_{i}.csv"))
                val_df = pd.read_csv(os.path.join(cv_path, f"val_fold_{i}.csv"))
                yield train_df, val_df
        elif os.path.exists(train_val_path):
            train_df = pd.read_csv(os.path.join(train_val_path, "train.csv"))
            val_df = pd.read_csv(os.path.join(train_val_path, "val.csv"))
            yield train_df, val_df

    def get_backtesting_fold_path(self, fold: int) -> dict:
        backtesting_path = self.get_backtesting_path()
        cv_path = os.path.join(backtesting_path, "cv")
        return {
            'train': os.path.join(cv_path, f"train_fold_{fold}.csv"),
            'val': os.path.join(cv_path, f"val_fold_{fold}.csv"),
        }

    def get_performance_estimation_fold(self):
        perf_path = self.get_performance_estimation_path()
        train_df = pd.read_csv(os.path.join(perf_path, "train.csv"))
        val_df = pd.read_csv(os.path.join(perf_path, "val.csv"))
        return train_df, val_df

    def get_final_model_fold(self):
        final_model_path = self.get_final_model_path()
        train_df = pd.read_csv(os.path.join(final_model_path, "train.csv"))
        val_df = pd.read_csv(os.path.join(final_model_path, "val.csv"))
        return train_df, val_df

    def get_test_set(self):
        return pd.read_csv(os.path.join(self.experiment_path, "test.csv"))

    def get_test_set_path(self) -> str:
        return os.path.join(self.experiment_path, "test.csv")
