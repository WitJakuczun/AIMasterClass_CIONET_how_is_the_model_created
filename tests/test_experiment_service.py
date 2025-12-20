import polars as pl
import pytest
from mlops.experiment_service import ExperimentService, RunType
from pathlib import Path
import os

class MockPaths:
    def __init__(self, experiments_path):
        self.experiments = experiments_path

@pytest.fixture
def experiment_env(tmp_path):
    exp_id = "exp_test"
    exp_path = tmp_path / exp_id
    exp_path.mkdir()
    
    # Create test.csv
    pl.DataFrame({"text": ["test"]}).write_csv(exp_path / "test.csv")
    
    # Create backtesting train-val files
    bt_path = exp_path / "backtesting" / "train-val"
    bt_path.mkdir(parents=True)
    pl.DataFrame({"text": ["train"]}).write_csv(bt_path / "train.csv")
    pl.DataFrame({"text": ["val"]}).write_csv(bt_path / "val.csv")
    pl.DataFrame({"text": ["test"]}).write_csv(bt_path / "test.csv")
    
    return exp_id, MockPaths(str(tmp_path))

def test_get_backtesting_folds(experiment_env):
    exp_id, paths = experiment_env
    service = ExperimentService(exp_id, paths)
    
    folds = list(service.get_backtesting_folds())
    assert len(folds) == 1
    train, val, test = folds[0]
    assert isinstance(train, pl.DataFrame)
    assert train["text"][0] == "train"

def test_get_train_val_split(experiment_env):
    exp_id, paths = experiment_env
    service = ExperimentService(exp_id, paths)
    
    # We need to create files for performance_estimation to test this
    pe_path = Path(paths.experiments) / exp_id / "performance_estimation"
    pe_path.mkdir()
    pl.DataFrame({"text": ["pe_train"]}).write_csv(pe_path / "train.csv")
    pl.DataFrame({"text": ["pe_val"]}).write_csv(pe_path / "val.csv")
    
    train, val = service.get_train_val_split(RunType.PERFORMANCE_ESTIMATION)
    assert isinstance(train, pl.DataFrame)
    assert train["text"][0] == "pe_train"

def test_get_test_set(experiment_env):
    exp_id, paths = experiment_env
    service = ExperimentService(exp_id, paths)
    
    test_set = service.get_test_set()
    assert isinstance(test_set, pl.DataFrame)
    assert test_set["text"][0] == "test"
