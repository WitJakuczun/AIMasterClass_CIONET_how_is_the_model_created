import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from loguru import logger
from pathlib import Path
import re
from tqdm import tqdm

from mlops.model_interface import ModelInterface

class PromptGemmaModel(ModelInterface):
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.prompt_template = None
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        logger.info(f"Using device: {self.device}")

    def _load_model(self):
        if self.model is None or self.tokenizer is None:
            if not self.model_name:
                raise ValueError("Model name not set. Was the train method called first?")
            
            logger.info(f"Loading base model for prompt-based prediction: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                dtype=torch.bfloat16
            )

    def train(self, train_dataset: pd.DataFrame, val_dataset: pd.DataFrame, hyperparameters: dict, output_dir: str):
        logger.info("PromptGemmaModel does not require training. Skipping.")
        self.model_name = hyperparameters.get('model_name')
        training_args = hyperparameters.get('training_arguments', {})
        self.prompt_template = training_args.get('prompt_template')

        if not self.model_name:
            raise ValueError("'model_name' not found in hyperparameters.")
        if not self.prompt_template:
            raise ValueError("'prompt_template' not found in training_arguments.")
        if "{text_to_analyze}" not in self.prompt_template:
            raise ValueError("Prompt template must contain '{text_to_analyze}' placeholder.")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory ensured at: {output_dir}")

    def predict(self, model_dir: str, data_to_predict: pd.DataFrame, output_dir: str):
        self._load_model()

        predictions = []

        for index, row in tqdm(data_to_predict.iterrows(), total=data_to_predict.shape[0], desc="Predicting"):
            text = row['Sentence']
            prompt = self.prompt_template.format(text_to_analyze=text)
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            response = self.model.generate(**inputs, max_new_tokens=5)
            output_text = self.tokenizer.decode(response[0], skip_special_tokens=True)
            
            generated_response = output_text.split(prompt)[-1].strip()

            match = re.search(r'(positive|negative|neutral)', generated_response, re.IGNORECASE)
            if match:
                sentiment = match.group(1).lower()
            else:
                print(f"{text=} => {generated_response=}")
                sentiment = "unknown"
            
            predictions.append(sentiment)

        output_df = data_to_predict.copy()
        output_df['Predicted_Sentiment'] = predictions
        output_path = Path(output_dir) / "predictions.csv"
        output_df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")