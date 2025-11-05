
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Paths:
    root: Path = Path(__file__).parent.resolve()
    artifacts: Path = root / "artifacts"
    data: Path = artifacts / "data"
    experiments: Path = artifacts / "experiments"
    trained_models: Path = artifacts / "trained_models"
    predictions: Path = artifacts / "predictions"
    metrics: Path = artifacts / "metrics"
    all_train_csv: Path = data / "all_train.csv"

paths = Paths()
