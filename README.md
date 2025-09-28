# Fine-tuning NLP models for sentiment analysis

This project provides a CLI application to fine-tune a series of NLP models for sentiment analysis. It uses a cross-validation strategy to train and evaluate the model. Currently the code covers the following models:
- Gemma3
- Roberta
- Bert-small
- Bert-micro
- SVM

## Workflow

The application workflow is designed around a clean separation of concerns, orchestrated by a central `Application` class.

1.  **Generate CV Splits**: Create cross-validation splits from the training data.
2.  **Generate Train/Validation/Test Split**: Create a single train/validation/test split.
3.  **Train**: Train the model on each fold of the cross-validation splits using a modular ML pipeline.
4.  **Predict**: Make predictions on the validation set of each fold using a modular ML pipeline.
5.  **Evaluate**: Evaluate the predictions, calculate metrics, and store them in a structured `results.json` file for persistent tracking.
6.  **Run Experiment**: Execute a full experiment, encompassing all the steps above, with comprehensive result tracking.


## Listing Available Models

To see a list of all available models that you can use for training, run the following command:

```bash
uv run -- python main.py list-models
```
*Note: Model configurations are now loaded dynamically from individual JSON files in the `model_configs/` directory.* 

## Step-by-step tutorial

### 1. Generate CV Splits

This step will create a new experiment directory, generate the cross-validation splits, and create a hold-out test set. The default number of splits is 5 and the default test set size is 20%.

```bash
uv run -- python main.py generate-cv-splits --experiment-id exp001 --test-size 0.2
```

**Expected result:**

A new directory `experiments/exp001` will be created with a `test.csv` file and 5 pairs of `train_fold_*.csv` and `val_fold_*.csv` files.

### 2. Generate Train/Validation/Test Split

This step will create a new experiment directory and generate a single train/validation/test split based on the provided ratios.

```bash
uv run -- python main.py generate-split --experiment-id exp002 --train-ratio 0.7 --val-ratio 0.15
```

**Expected result:**

A new directory `experiments/exp002` will be created with `train.csv`, `val.csv`, and `test.csv` files.

### 3. Train the model

This step will train the model on a specific fold of the experiment. You need to specify the experiment ID and the fold number. The training process now utilizes a modular ML pipeline.

```bash
uv run -- python main.py train --experiment-id exp001 --fold-number 0
```

**Expected result:**

A new directory `trained_models/exp001_fold_0` will be created with the trained model.

### 4. Make predictions

This step will make predictions on the validation set of a specific fold. You need to specify the experiment ID and the fold number. Predictions are generated via a modular ML pipeline.

```bash
uv run -- python main.py predict --experiment-id exp001 --fold-number 0
```

**Expected result:**

A new directory `predictions/exp001_fold_0` will be created with a `predictions.csv` file.

### 5. Evaluate the predictions

This step will evaluate the predictions for a specific fold, calculate metrics, and **persistently store them in `experiments/exp001/results.json`**.

```bash
uv run -- python main.py evaluate --experiment-id exp001 --fold-number 0
```

**Expected result:**

The metrics for the specified fold will be logged to the console and saved to `experiments/exp001/results.json`.

### 6. Run a full experiment

This step will run the full experiment, which includes generating the CV splits, training the model on each fold, making predictions, and evaluating the results. All metrics are tracked and saved.

```bash
uv run -- python main.py run-experiment --experiment-id exp001
```

**Expected result:**

The full experiment will be executed. A detailed summary table of metrics for each fold will be printed to the console, and all results will be saved to `experiments/exp001/results.json`.