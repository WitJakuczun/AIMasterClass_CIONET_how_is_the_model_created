
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
from loguru import logger
import sys
from packaging import version
import transformers
from src.model_interface import ModelInterface

class GemmaModel(ModelInterface):
    """
    A wrapper for the Gemma model that implements the ModelInterface.
    """

    def train(self, train_dataset: pd.DataFrame, hyperparameters: dict, output_dir: str, val_dataset: pd.DataFrame = None):
        """
        Trains a Gemma model on the provided dataset.

        Args:
            train_dataset: A pandas DataFrame containing the training data. 
                           It is expected to have 'Sentence' and 'Sentiment' columns.
            hyperparameters: A dictionary of hyperparameters for training.
            output_dir: The directory where the trained model will be saved.
            val_dataset: An optional pandas DataFrame containing the validation data.
        """

        # --- 0. Setup Logging with Loguru ---
        os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, "training.log")

        logger.remove()
        logger.add(sys.stderr, level="INFO")
        logger.add(log_file, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}")
        
        logger.info(f"Logging configured. Log file: {log_file}")

        # --- 1. Load Tokenizer and Model ---
        logger.info("Loading tokenizer and model...")
        model_name = hyperparameters.get("model_name", "google/gemma-2b")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation='eager')

        # Add a padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # type: ignore
            model.resize_token_embeddings(len(tokenizer))
            logger.info("Added padding token to tokenizer.")

        # --- 2. Prepare the Dataset ---
        logger.info("Preparing dataset...")
        def format_row(row):
            return f"Sentence: {row['Sentence']} Sentiment: {row['Sentiment']}"

        text_data = train_dataset.apply(format_row, axis=1).tolist()
        hf_dataset = Dataset.from_dict({'text': text_data})

        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

        tokenized_dataset = hf_dataset.map(tokenize_function, batched=True)
        tokenized_dataset = tokenized_dataset.map(lambda examples: {'labels': examples['input_ids']}, batched=True)
        tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        logger.info("Dataset prepared and tokenized.")

        eval_dataset = None
        if val_dataset is not None:
            logger.info("Preparing validation dataset...")
            val_text_data = val_dataset.apply(format_row, axis=1).tolist()
            val_hf_dataset = Dataset.from_dict({'text': val_text_data})
            eval_dataset = val_hf_dataset.map(tokenize_function, batched=True)
            eval_dataset = eval_dataset.map(lambda examples: {'labels': examples['input_ids']}, batched=True)
            eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
            logger.info("Validation dataset prepared and tokenized.")

        # --- 3. Configure Training ---
        logger.info("Configuring training arguments...")
        default_training_args = {
            "output_dir": output_dir,
            "num_train_epochs": 1,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "fp16": torch.cuda.is_available(),
            "logging_steps": 10,
            "save_total_limit": 2,
            "report_to": "none", # Disable default reporting, as we handle logging
            "dataloader_pin_memory": False
        }

        if eval_dataset is not None:
            if version.parse(transformers.__version__) >= version.parse("4.6.0"):
                default_training_args["evaluation_strategy"] = "steps"
                default_training_args["eval_steps"] = 10
            else:
                logger.warning("'evaluation_strategy' not supported in this version of transformers. Skipping evaluation.")

        training_args_dict = {**default_training_args, **hyperparameters.get("training_arguments", {})}
        training_arguments = TrainingArguments(**training_args_dict)

        # --- 4. Instantiate and Run Trainer ---
        trainer = Trainer(
            model=model,
            args=training_arguments,
            train_dataset=tokenized_dataset,
            eval_dataset=eval_dataset,
        )

        logger.info("Starting model training...")
        trainer.train()
        logger.info("Training finished.")

        # --- 5. Save the Model ---
        logger.info(f"Saving model to {output_dir}...")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info("Model and tokenizer saved successfully.")

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
        logger.add(log_file, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}")

        logger.info(f"Logging configured. Log file: {log_file}")

        # --- 1. Load Tokenizer and Model ---
        logger.info(f"Loading model from {model_dir}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = AutoModelForCausalLM.from_pretrained(model_dir)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        logger.info(f"Model loaded successfully on device: {device}")

        # --- 2. Generate Predictions ---
        predictions = []
        logger.info(f"Generating predictions for {len(data_to_predict)} sentences...")

        for index, row in data_to_predict.iterrows():
            sentence = row['Sentence']
            prompt = f"Sentence: {sentence} Sentiment:"
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Generate text
            # The max_new_tokens is set to a small number to just get the sentiment
            outputs = model.generate(**inputs, max_new_tokens=5, pad_token_id=tokenizer.pad_token_id)
            
            # Decode the output
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the sentiment
            # The generated text will be in the format "Sentence: [sentence] Sentiment: [sentiment]"
            try:
                predicted_sentiment = generated_text.split("Sentiment:")[1].strip()
                predictions.append(predicted_sentiment)
                logger.info(f'  - Predicted {predicted_sentiment} for: {sentence[:50]}...')
            except IndexError:
                logger.warning(f'  - Could not extract sentiment for: "{sentence[:50]}..."')
                predictions.append(None)

        # --- 3. Save Predictions ---
        logger.info("Saving predictions...")
        output_df = data_to_predict.copy()
        output_df['Predicted_Sentiment'] = predictions
        
        output_file = os.path.join(output_dir, "predictions.csv")
        output_df.to_csv(output_file, index=False)
        logger.info(f"Predictions saved to {output_file}")
