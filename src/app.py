
from src import runner
from src.generate_CV_splits import generate_cv_splits, generate_split

class Application:
    def __init__(self, runner_module):
        self.runner = runner_module

    def run_experiment(self, experiment_id: str, model_config_name: str, model_output_dir: str, prediction_output_dir: str):
        self.runner.run_experiment(experiment_id, model_config_name, model_output_dir, prediction_output_dir)

    def train_model(self, experiment_id: str, fold_number: int, model_config_name: str, model_output_dir: str):
        self.runner.train_model(experiment_id, fold_number, model_config_name, model_output_dir)

    def predict_model(self, experiment_id: str, fold_number: int, model_config_name: str, model_input_dir: str, prediction_output_dir: str):
        self.runner.predict_model(experiment_id, fold_number, model_config_name, model_input_dir, prediction_output_dir)

    def evaluate_predictions(self, experiment_id: str, fold_number: int, prediction_input_dir: str):
        return self.runner.evaluate_predictions(experiment_id, fold_number, prediction_input_dir)

    def generate_cv_splits(self, n_splits: int, experiment_id: str, input_file: str, target_column: str, test_size: float):
        generate_cv_splits(n_splits, experiment_id, input_file, target_column, test_size)

    def generate_split(self, experiment_id: str, input_file: str, target_column: str, train_ratio: float, val_ratio: float):
        generate_split(experiment_id, input_file, target_column, train_ratio, val_ratio)
