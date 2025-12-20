import os
import polars as pl
from dataclasses import dataclass, field
from typing import List, Tuple
import glob
from loguru import logger


# --- DataClass for Experiment ---
@dataclass
class CVExperiment:
    experiment_id: str
    path: str
    folds: List[Tuple[pl.DataFrame, pl.DataFrame]] = field(
        default_factory=list, repr=False
    )

    def __post_init__(self):
        if not self.folds:
            if self.is_cv_experiment:
                train_files = sorted(
                    glob.glob(os.path.join(self.path, "train_fold_*.csv"))
                )
                for f in train_files:
                    val_f = f.replace("train_fold_", "val_fold_")
                    if os.path.exists(val_f):
                        self.folds.append((pl.read_csv(f), pl.read_csv(val_f)))
            else:
                train_path = os.path.join(self.path, "train.csv")
                val_path = os.path.join(self.path, "val.csv")
                if os.path.exists(train_path) and os.path.exists(val_path):
                    self.folds.append((pl.read_csv(train_path), pl.read_csv(val_path)))

    @property
    def is_cv_experiment(self) -> bool:
        """Checks if the experiment is a cross-validation experiment."""
        train_files = glob.glob(os.path.join(self.path, "train_fold_*.csv"))
        return len(train_files) > 0

    def get_fold(self, fold_number: int) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Returns the train and validation dataframes for a given fold."""
        if 0 <= fold_number < len(self.folds):
            return self.folds[fold_number]
        else:
            raise IndexError(f"Fold number {fold_number} is out of range.")

    @property
    def n_splits(self) -> int:
        """Returns the number of splits in the experiment."""
        return len(self.folds)
