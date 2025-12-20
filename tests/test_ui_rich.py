
import pytest

from config import paths
from mlops.app import Application
from models_config import MODELS


@pytest.fixture
def app():
    return Application(paths, MODELS)


def test_list_models_rich(app):
    # Just check if it runs without error
    try:
        app.list_models()
    except Exception as e:
        pytest.fail(f"list_models failed with Rich enhancement: {e}")


def test_list_experiments_rich(app):
    # Just check if it runs without error
    try:
        app.list_experiments()
    except Exception as e:
        pytest.fail(f"list_experiments failed with Rich enhancement: {e}")
