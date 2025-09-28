
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Paths:
    root: Path = Path(__file__).parent.resolve()
    data: Path = root / "data"
    experiments: Path = root / "experiments"
    trained_models: Path = root / "trained_models"
    predictions: Path = root / "predictions"
    all_train_csv: Path = data / "all_train.csv"

paths = Paths()
