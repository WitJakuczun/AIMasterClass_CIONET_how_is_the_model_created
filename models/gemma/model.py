
import pandas as pd
import os
from loguru import logger
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import torch

from mlops.model_interface import ModelInterface
from mlops.prompt_engineering import create_gemma_prompt

class GemmaModel(ModelInterface):
    """
    A wrapper for the Gemma model that implements the ModelInterface.
    """

    def train(self, train_dataset: pd.DataFrame, hyperparameters: dict, output_dir: str, val_dataset: pd.DataFrame = None):
        """
        Trains a Gemma model.

        Args:
            train_dataset: A pandas DataFrame containing the training data.
            hyperparameters: A dictionary of hyperparameters for the Gemma model.
            output_dir: The directory where the trained model will be saved.
            val_dataset: A pandas DataFrame containing the validation data.
        """

        # --- 0. Setup Logging ---
        os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, "training.log")

        logger.remove()
        logger.add(sys.stderr, level="INFO")
        logger.add(log_file, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}")

        logger.info(f"Logging configured. Log file: {log_file}")

        # --- 1. Load Model and Tokenizer ---
        model_name = hyperparameters.get('model_name', 'google/gemma-2b-it')
        logger.info(f"Loading model and tokenizer for {model_name}...")

        quantization_config = BitsAndBytesConfig(**hyperparameters.get('quantization_config', {}))
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            token=True
        )

        # --- 2. Prepare model for training ---
        logger.info("Preparing model for k-bit training...")
        model = prepare_model_for_kbit_training(model)

        # --- 3. Configure LoRA ---
        logger.info("Configuring LoRA...")
        lora_config = LoraConfig(**hyperparameters.get('lora_config', {}))
        model = get_peft_model(model, lora_config)

        # --- 4. Prepare Datasets ---
        logger.info("Preparing datasets...")
        train_dataset['prompt'] = train_dataset.apply(create_gemma_prompt, axis=1)
        if val_dataset is not None:
            val_dataset['prompt'] = val_dataset.apply(create_gemma_prompt, axis=1)

        # --- 5. Configure Training Arguments ---
        logger.info("Configuring training arguments...")
        training_args_dict = hyperparameters.get('training_arguments', {})
        training_args_dict['output_dir'] = output_dir
        training_arguments = TrainingArguments(**training_args_dict)

        # --- 6. Create Trainer ---
        logger.info("Creating SFTTrainer...")
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            peft_config=lora_config,
            dataset_text_field="prompt",
            max_seq_length=hyperparameters.get('max_seq_length', 512),
            tokenizer=tokenizer,
            args=training_arguments,
            packing=False,
        )

        # --- 7. Train the Model ---
        logger.info("Starting training...")
        trainer.train()
        logger.info("Training finished.")

        # --- 8. Save the Model ---
        logger.info(f"Saving model to {output_dir}...")
        trainer.save_model(output_dir)
        logger.info(f"Model saved successfully to {output_dir}.")

    def predict(self, model_dir: str, data_to_predict: pd.DataFrame, output_dir: str):
        """
        Generates predictions using a trained Gemma model.

        Args:
            model_dir: The directory where the trained model is saved.
            data_to_predict: A pandas DataFrame with a 'Sentence' column.
            output_dir: The directory where the predictions will be saved.
        """

        # --- 0. Setup Logging ---
        os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, "prediction.log")

        logger.remove()
        logger.add(sys.stderr, level="INFO")
        logger.add(log_file, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}")

        logger.info(f"Logging configured. Log file: {log_file}")

        # --- 1. Load Model and Tokenizer ---
        logger.info(f"Loading model and tokenizer from {model_dir}...")
        try:
            model = AutoModelForCausalLM.from_pretrained(model_dir)
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return
        logger.info("Model and tokenizer loaded successfully.")

        # --- 2. Generate Predictions ---
        logger.info(f"Generating predictions for {len(data_to_predict)} sentences...")
        
        def get_prediction(text):
            prompt = f"<start_of_turn>user\nOceń sentyment poniższego zdania jako negatywny, neutralny lub pozytywny. Odpowiedz tylko jednym słowem.\nZdanie: {text}<end_of_turn>\n<start_of_turn>model\n"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=5) # Limit tokens to get just the sentiment
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract the answer after the model tag
            return result.split("model\n")[-1].strip()

        data_to_predict['Predicted_Sentiment'] = data_to_predict['Sentence'].apply(get_prediction)

        # --- 3. Save Predictions ---
        logger.info("Saving predictions...")
        output_file = os.path.join(output_dir, "predictions.csv")
        data_to_predict.to_csv(output_file, index=False)
        logger.info(f"Predictions saved to {output_file}")
