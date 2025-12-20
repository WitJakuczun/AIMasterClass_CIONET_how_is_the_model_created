import polars as pl
import pytest
from mlops.splitting import DataSplitter

@pytest.fixture
def sample_df():
    return pl.DataFrame({
        "text": ["a", "b", "c", "d", "e", "f", "g", "h"],
        "Sentiment": ["pos", "pos", "pos", "pos", "neg", "neg", "neg", "neg"]
    })

def test_split_train_test(sample_df):
    splitter = DataSplitter()
    train, test = splitter.split_train_test(sample_df, test_size=0.25, stratify_on="Sentiment")
    
    assert isinstance(train, pl.DataFrame)
    assert isinstance(test, pl.DataFrame)
    assert len(train) == 6
    assert len(test) == 2
    # Check stratification (simple case)
    assert test["Sentiment"].value_counts().filter(pl.col("Sentiment") == "pos")["count"][0] == 1
    assert test["Sentiment"].value_counts().filter(pl.col("Sentiment") == "neg")["count"][0] == 1

def test_split_cv(sample_df):
    splitter = DataSplitter()
    folds = list(splitter.split_cv(sample_df, n_splits=2, stratify_on="Sentiment"))
    
    assert len(folds) == 2
    for fold, train, val, test in folds:
        assert isinstance(train, pl.DataFrame)
        assert isinstance(val, pl.DataFrame)
        assert isinstance(test, pl.DataFrame)
        assert len(train) == 4
        assert len(val) == 2
        assert len(test) == 2

def test_split_train_val(sample_df):
    splitter = DataSplitter()
    train, val, test = splitter.split_train_val(sample_df, val_size=0.25, test_size=0.25, stratify_on="Sentiment")
    
    assert len(train) == 4
    assert len(val) == 2
    assert len(test) == 2
    assert isinstance(train, pl.DataFrame)
    assert isinstance(val, pl.DataFrame)
    assert isinstance(test, pl.DataFrame)
