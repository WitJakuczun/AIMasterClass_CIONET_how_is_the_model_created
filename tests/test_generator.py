import polars as pl
import pytest
from unittest.mock import MagicMock, patch
from data.generator import DataGenerator
import os


@pytest.fixture
def mock_generator():
    with (
        patch("transformers.AutoTokenizer.from_pretrained"),
        patch("transformers.AutoModelForCausalLM.from_pretrained"),
    ):
        return DataGenerator()


def test_determine_generation_counts_fixed(mock_generator):
    counts = mock_generator._determine_generation_counts(10, None, None)
    assert counts == {"positive": 10, "neutral": 10, "negative": 10}


def test_determine_generation_counts_from_file(mock_generator, tmp_path):
    # Create a dummy CSV file with Polars
    df = pl.DataFrame({"Sentiment": ["positive", "positive", "neutral", "negative"]})
    csv_path = tmp_path / "dist.csv"
    df.write_csv(csv_path)

    counts = mock_generator._determine_generation_counts(None, str(csv_path), 100)
    # Distribution: pos: 0.5, neu: 0.25, neg: 0.25
    # Expected: pos: 50, neu: 25, neg: 25
    assert counts == {"positive": 50, "neutral": 25, "negative": 25}


def test_determine_generation_counts_rounding(mock_generator, tmp_path):
    # Distribution: 2 positive, 1 neutral -> 2/3 and 1/3
    df = pl.DataFrame({"Sentiment": ["positive", "positive", "neutral"]})
    csv_path = tmp_path / "dist_round.csv"
    df.write_csv(csv_path)

    counts = mock_generator._determine_generation_counts(None, str(csv_path), 10)
    # 10 * 0.666 = 6.66 -> 6
    # 10 * 0.333 = 3.33 -> 3
    # Total = 9. Remainder 1 added to 'positive' (most frequent)
    # Expected: pos: 7, neu: 3
    assert counts["positive"] == 7
    assert counts["neutral"] == 3
    assert sum(counts.values()) == 10
