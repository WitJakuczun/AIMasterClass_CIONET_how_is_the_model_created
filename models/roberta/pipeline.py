
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from loguru import logger
from sklearn.metrics import accuracy_score
from mlops.pipeline import PipelineStep, PipelineContext
from packaging import version
import transformers

class RobertaDataPreprocessor(PipelineStep):
    def run(self, context: PipelineContext) -> PipelineContext:
        logger.info("Preprocessing data for RoBERTa model...")
        train_dataset = context.get("train_dataset")
        val_dataset = context.get("val_dataset")
        hyperparameters = context.get("hyperparameters")
        
        labels = train_dataset['Sentiment'].unique().tolist()
        label2id = {label: i for i, label in enumerate(labels)}
        id2label = {i: label for i, label in enumerate(labels)}
        
        train_dataset['label'] = train_dataset['Sentiment'].map(label2id)
        
        model_name = hyperparameters.get("model_name", "roberta-base")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        hf_dataset = Dataset.from_pandas(train_dataset)

        def tokenize_function(examples):
            return tokenizer(examples["Sentence"], padding="max_length", truncation=True, max_length=512)

        tokenized_dataset = hf_dataset.map(tokenize_function, batched=True)
        tokenized_dataset = tokenized_dataset.remove_columns([col for col in train_dataset.columns if col not in ['label', 'input_ids', 'attention_mask']])
        tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        
        context.set("tokenized_dataset", tokenized_dataset)

        if val_dataset is not None:
            logger.info("Preprocessing validation data...")
            val_dataset['label'] = val_dataset['Sentiment'].map(label2id)
            hf_val_dataset = Dataset.from_pandas(val_dataset)
            tokenized_val_dataset = hf_val_dataset.map(tokenize_function, batched=True)
            tokenized_val_dataset = tokenized_val_dataset.remove_columns([col for col in val_dataset.columns if col not in ['label', 'input_ids', 'attention_mask']])
            tokenized_val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
            context.set("tokenized_val_dataset", tokenized_val_dataset)
            logger.info("Validation data preprocessing complete.")

        context.set("tokenizer", tokenizer)
        context.set("label2id", label2id)
        context.set("id2label", id2label)
        logger.info("Data preprocessing complete.")
        return context

class RobertaTrainer(PipelineStep):
    def run(self, context: PipelineContext) -> PipelineContext:
        logger.info("Training RoBERTa model...")
        output_dir = context.get("output_dir")
        hyperparameters = context.get("hyperparameters")
        tokenized_dataset = context.get("tokenized_dataset")
        tokenized_val_dataset = context.get("tokenized_val_dataset")
        tokenizer = context.get("tokenizer")
        label2id = context.get("label2id")
        id2label = context.get("id2label")

        model_name = hyperparameters.get("model_name", "roberta-base")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(label2id),
            label2id=label2id,
            id2label=id2label
        )

        default_training_args = {
            "output_dir": output_dir,
            "num_train_epochs": 1,
            "per_device_train_batch_size": 8,
            "gradient_accumulation_steps": 1,
            "learning_rate": 2e-5,
            "fp16": torch.cuda.is_available(),
            "logging_steps": 10,
            "save_total_limit": 2,
            "report_to": "none",
            "dataloader_pin_memory": False
        }

        if tokenized_val_dataset is not None:
            if version.parse(transformers.__version__) >= version.parse("4.6.0"):
                default_training_args["eval_strategy"] = "steps"
                default_training_args["eval_steps"] = 10
            else:
                logger.warning("'evaluation_strategy' not supported in this version of transformers. Skipping evaluation.")

        training_args_dict = {**default_training_args, **hyperparameters.get("training_arguments", {})}
        training_arguments = TrainingArguments(**training_args_dict)

        def compute_metrics(p):
            predictions, labels = p
            predictions = predictions.argmax(axis=1)
            return {"accuracy": accuracy_score(labels, predictions)}

        trainer = Trainer(
            model=model,
            args=training_arguments,
            train_dataset=tokenized_dataset,
            eval_dataset=tokenized_val_dataset,
            compute_metrics=compute_metrics
        )

        trainer.train()
        
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Model and tokenizer saved to {output_dir}")
        return context

class RobertaPredictor(PipelineStep):
    def run(self, context: PipelineContext) -> PipelineContext:
        logger.info("Running RoBERTa predictions...")
        model_dir = context.get("model_dir")
        data_to_predict = context.get("data_to_predict")
        output_file = context.get("output_file")

        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        id2label = model.config.id2label
        predictions = []

        for index, row in data_to_predict.iterrows():
            sentence = row['Sentence']
            inputs = tokenizer(sentence, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            logits = outputs.logits
            predicted_class_id = torch.argmax(logits, dim=1).item()
            predicted_sentiment = id2label[predicted_class_id]
            predictions.append(predicted_sentiment)

        output_df = data_to_predict.copy()
        output_df['Predicted_Sentiment'] = predictions
        
        output_df.to_csv(output_file, index=False)
        logger.info(f"Predictions saved to {output_file}")
        return context
