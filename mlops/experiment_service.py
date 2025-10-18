

from pathlib import Path
import pandas as pd
from loguru import logger
import glob
from enum import Enum

class RunType(str, Enum):
    BACKTESTING = "backtesting"
    PERFORMANCE_ESTIMATION = "performance_estimation"
    FINAL_MODEL = "final_model"

class ExperimentService:
    def __init__(self, experiment_id: str, paths_config: object):
        self.experiment_id = experiment_id
        self.experiment_path = Path(paths_config.experiments) / self.experiment_id

    def get_run_path(self, run_type: RunType) -> Path:
        return self.experiment_path / run_type.value

    def get_backtesting_folds(self):
        backtesting_path = self.get_run_path(RunType.BACKTESTING)
        cv_path = backtesting_path / "cv"
        train_val_path = backtesting_path / "train-val"

        if cv_path.exists():
            fold_files = glob.glob(str(cv_path / "train_fold_*.csv"))
            for i in range(len(fold_files)):
                train_df = pd.read_csv(cv_path / f"train_fold_{i}.csv")
                val_df = pd.read_csv(cv_path / f"val_fold_{i}.csv")
                yield train_df, val_df
        elif train_val_path.exists():
            train_df = pd.read_csv(train_val_path / "train.csv")
            val_df = pd.read_csv(train_val_path / "val.csv")
            yield train_df, val_df

    def get_backtesting_fold_path(self, fold: int) -> dict:
        backtesting_path = self.get_run_path(RunType.BACKTESTING)
        cv_path = backtesting_path / "cv"
        train_val_path = backtesting_path / "train-val"

        if cv_path.exists():
            return {
                'train': str(cv_path / f"train_fold_{fold}.csv"),
                'val': str(cv_path / f"val_fold_{fold}.csv"),
            }
        elif train_val_path.exists():
            return {
                'train': str(train_val_path / "train.csv"),
                'val': str(train_val_path / "val.csv"),
            }
        else:
            raise FileNotFoundError(f"No backtesting data found in {backtesting_path}")

    def get_train_val_split(self, run_type: RunType) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Gets the train/val split for performance estimation or final model training."""
        run_path = self.get_run_path(run_type)
        train_df = pd.read_csv(run_path / "train.csv")
        val_df = pd.read_csv(run_path / "val.csv")
        return train_df, val_df

    def get_test_set(self) -> pd.DataFrame:
        return pd.read_csv(self.experiment_path / "test.csv")

    def get_test_set_path(self) -> str:
        return str(self.experiment_path / "test.csv")
