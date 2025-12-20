from mlops import setup_logging
from loguru import logger
from rich.logging import RichHandler
import pytest

def test_setup_logging_configures_rich():
    """
    Verifies that setup_logging adds a RichHandler to loguru.
    """
    # Setup logging
    try:
        setup_logging()
    except Exception as e:
        pytest.fail(f"setup_logging failed: {e}")
    
    # Check handlers
    handlers = logger._core.handlers
    has_rich = False
    for h in handlers.values():
        sink = h._sink
        # Unpack if wrapped (StandardSink)
        if hasattr(sink, "_handler"):
             sink = sink._handler
        
        if "RichHandler" in str(type(sink).__name__) or isinstance(sink, RichHandler):
             has_rich = True
             break
    
    assert has_rich, "RichHandler was not found in loguru handlers"
