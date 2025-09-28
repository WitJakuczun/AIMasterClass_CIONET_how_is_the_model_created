
import os
import json
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class ModelConfig:
    model_path: str
    model_name: Optional[str] = None
    training_arguments: Dict[str, Any] = field(default_factory=dict)

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
                
                # Ensure all required fields are present, provide defaults for optional ones
                model_path = config_data.get("model_path")
                if not model_path:
                    raise ValueError(f"model_path missing in {config_file}")

                model_name = config_data.get("model_name")
                training_arguments = config_data.get("training_arguments", {})

                MODELS[config_name] = ModelConfig(
                    model_path=model_path,
                    model_name=model_name,
                    training_arguments=training_arguments
                )
            except (IOError, json.JSONDecodeError, ValueError) as e:
                print(f"Error loading model configuration from {config_file}: {e}")
else:
    print(f"Warning: Model configurations directory '{MODEL_CONFIGS_DIR}' not found.")
