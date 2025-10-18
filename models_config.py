
import os
import json
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, FilePath

class HyperparameterSearch(BaseModel):
    n_iter: int = Field(..., gt=0)
    param_distributions: Dict[str, Any]

class ModelConfig(BaseModel):
    module_path: str
    class_name: str
    model_path: Optional[FilePath] = None
    model_name: Optional[str] = None
    training_arguments: Dict[str, Any] = {}
    hyperparameter_search: Optional[HyperparameterSearch] = None

MODELS: Dict[str, ModelConfig] = {}

# Dynamically load model configurations from the 'model_configs' directory
MODEL_CONFIGS_DIR = "model_configs"

if os.path.exists(MODEL_CONFIGS_DIR):
    for config_file in os.listdir(MODEL_CONFIGS_DIR):
        if config_file.endswith(".json"):
            config_name = os.path.splitext(config_file)[0]
            config_path = os.path.join(MODEL_CONFIGS_DIR, config_file)
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                
                MODELS[config_name] = ModelConfig(**config_data)
            except (IOError, json.JSONDecodeError, ValueError) as e:
                print(f"Error loading model configuration from {config_file}: {e}")
else:
    print(f"Warning: Model configurations directory '{MODEL_CONFIGS_DIR}' not found.")
