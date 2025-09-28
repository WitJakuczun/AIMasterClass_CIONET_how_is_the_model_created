
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import joblib
from loguru import logger
import sys
from src.model_interface import ModelInterface

class SVMModel(ModelInterface):
    """
    A wrapper for the SVM model that implements the ModelInterface.
    """

    def train(self, train_dataset: pd.DataFrame, hyperparameters: dict, output_dir: str):
        """
        Trains an SVM model with CountVectorizer.

        Args:
            train_dataset: A pandas DataFrame containing the training data.
                           It is expected to have 'Sentence' and 'Sentiment' columns.
            hyperparameters: A dictionary of hyperparameters for the SVM model.
            output_dir: The directory where the trained model will be saved.
        """

        # --- 0. Setup Logging ---
        os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, "training.log")

        logger.remove()
        logger.add(sys.stderr, level="INFO")
        logger.add(log_file, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}")

        logger.info(f"Logging configured. Log file: {log_file}")

        # --- 1. Create the pipeline ---
        logger.info("Creating SVM pipeline...")
        
        # Get hyperparameters for SVC
        svc_hyperparameters = hyperparameters.get('training_arguments', {})
        
        pipeline = Pipeline([
            ('vect', CountVectorizer()),
            ('clf', SVC(**svc_hyperparameters)),
        ])

        # --- 2. Train the model ---
        logger.info("Training the SVM model...")
        X_train = train_dataset['Sentence']
        y_train = train_dataset['Sentiment']
        
        pipeline.fit(X_train, y_train)
        logger.info("Training finished.")

        # --- 3. Save the Model ---
        logger.info(f"Saving model to {output_dir}...")
        model_path = os.path.join(output_dir, "svm_model.joblib")
        joblib.dump(pipeline, model_path)
        logger.info(f"Model saved successfully to {model_path}.")

    def predict(self, model_dir: str, data_to_predict: pd.DataFrame, output_dir: str):
        """
        Generates predictions using a trained SVM model.

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

        # --- 1. Load Model ---
        model_path = os.path.join(model_dir, "svm_model.joblib")
        logger.info(f"Loading model from {model_path}...")
        try:
            pipeline = joblib.load(model_path)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return
        logger.info("Model loaded successfully.")

        # --- 2. Generate Predictions ---
        logger.info(f"Generating predictions for {len(data_to_predict)} sentences...")
        X_to_predict = data_to_predict['Sentence']
        predictions = pipeline.predict(X_to_predict)

        # --- 3. Save Predictions ---
        logger.info("Saving predictions...")
        output_df = data_to_predict.copy()
        output_df['Predicted_Sentiment'] = predictions
        
        output_file = os.path.join(output_dir, "predictions.csv")
        output_df.to_csv(output_file, index=False)
        logger.info(f"Predictions saved to {output_file}")
