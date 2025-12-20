import os
from pathlib import Path
import json
import pandas as pd
from mlops.app import Application
from mlops.result_store import ResultStore
from loguru import logger

# Mock config classes
class MockConfig:
    def __init__(self, experiments_path):
        self.experiments = experiments_path

def reproduce():
    # Setup paths
    base_path = Path("temp_reproduce")
    experiments_path = base_path / "experiments"
    experiments_path.mkdir(parents=True, exist_ok=True)
    
    exp_id = "exp_test_viz"
    run_type = "backtesting"
    exp_dir = experiments_path / exp_id
    run_dir = exp_dir / run_type
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dummy results.json using ResultStore
    store = ResultStore(run_dir)
    metrics = {"f1_score": 0.85, "accuracy": 0.90}
    store.save_metrics(fold=0, model_name="model_a", metrics=metrics)
    metrics_b = {"f1_score": 0.88, "accuracy": 0.92}
    store.save_metrics(fold=0, model_name="model_b", metrics=metrics_b)
    
    # Mock Application
    app = Application(MockConfig(str(experiments_path)), {})
    
    # Attempt visualization with DEFAULT metric (which is f1_macro currently)
    logger.info("Attempting visualization...")
    result = app.visualize_comparison(exp_id, run_type) # Removed metric="f1_macro" to test default
    print(f"Result: {result}")

    # Check if failed as expected (before fix) or succeeded (after fix)
    if "Wykres został wygenerowany" in result:
        print("SUCCESS: Visualization generated.")
        # Check if file exists
        output_path = exp_dir / "visualizations" / f"comparison_{run_type}_f1_score.png"
        if output_path.exists():
            print(f"File found at {output_path}")
        else:
            print("ERROR: Output string says success but file missing.")
    else:
        print("FAILURE: Visualization failed as expected (or unexpected error).")

    # Cleanup
    import shutil
    if base_path.exists():
        shutil.rmtree(base_path)

if __name__ == "__main__":
    reproduce()
