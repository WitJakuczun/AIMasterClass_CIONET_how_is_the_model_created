
import os
import pandas as pd
from src.experiment import CVExperiment

def test_cvexperiment_loads_non_cv_data():
    """Tests that CVExperiment can load a simple train/val split."""
    exp_id = "exp001"
    base_path = os.path.abspath("tests/test_exp")
    exp_path = os.path.join(base_path, exp_id)
    
    experiment = CVExperiment(experiment_id=exp_id, path=exp_path)
    
    assert not experiment.is_cv_experiment
    assert experiment.n_splits == 1
    
    train_df, val_df = experiment.get_fold(0)
    
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert len(train_df) == 1
    assert len(val_df) == 1
    assert train_df.iloc[0]['col1'] == 'a'
    assert val_df.iloc[0]['col1'] == 'c'
