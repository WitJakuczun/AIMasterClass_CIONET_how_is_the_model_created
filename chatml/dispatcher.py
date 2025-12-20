# chatml/dispatcher.py
import contextlib
import io
import sys
from datetime import datetime
from pathlib import Path

from config import Paths
from mlops.app import Application
from models_config import MODELS


class Tee(io.TextIOBase):
    """Klasa pomocnicza, która kieruje wyjście do dwóch strumieni jednocześnie:
    1. Oryginalnego strumienia (np. stdout) - dla użytkownika.
    2. Bufora w pamięci - do przechwycenia dla Gemini.
    """

    def __init__(self, original_stream, capture_stream):
        self.original_stream = original_stream
        self.capture_stream = capture_stream

    def write(self, data):
        self.original_stream.write(data)
        self.capture_stream.write(data)
        # Wymuszamy flush, aby użytkownik widział logi na bieżąco
        self.original_stream.flush()

    def flush(self):
        self.original_stream.flush()
        self.capture_stream.flush()


class FunctionDispatcher:
    """Klasa odpowiedzialna za mapowanie i wywoływanie funkcji MLOps
    na podstawie odpowiedzi od Gemini.
    """

    def __init__(self):
        # Inicjalizacja głównej klasy aplikacyjnej z istniejącej logiki MLOps
        self.app = Application(paths_config=Paths, models_config=MODELS)
        self.function_map = {
            "generate_splits": self.app.generate_splits,
            "run_backtesting": self.app.run_backtesting,
            "train_final_model": self.app.train_final_model,
            "predict_new": self._predict_new_wrapper,
            "list_experiments": self.app.list_experiments,
            "list_models": self.app.list_models,
            "compare_models": self.app.compare_models,
            "estimate_performance": self.app.estimate_performance,
            "evaluate_prediction": self._evaluate_prediction_wrapper,
            "visualize_comparison": self.app.visualize_comparison,
        }

    def _predict_new_wrapper(self, model_path: str, input_file: str, name: str):
        """Wrapper dla predict_new, który generuje brakujące identyfikatory i znaczniki czasu.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prediction_id = f"pred_{timestamp}"

        return self.app.predict_new(
            model_path, input_file, prediction_id, timestamp, name
        )

    def _evaluate_prediction_wrapper(
        self, predictions_file: str, ground_truth_file: str, display: bool = True
    ):
        """Wrapper dla evaluate_prediction, który wyciąga metadane ze ścieżki pliku.
        """
        predictions_path = Path(predictions_file)
        # Zakładamy strukturę katalogów: .../timestamp/name/model_id/predictions.csv
        # Jeśli struktura jest inna, może to wymagać dostosowania
        try:
            model_id = predictions_path.parent.parent.name
            name = predictions_path.parent.parent.parent.name
            timestamp = predictions_path.parent.parent.parent.parent.name
        except Exception:
            # Fallback dla nietypowych ścieżek - użyj placeholderów
            model_id = "unknown_model"
            name = "unknown_name"
            timestamp = "unknown_timestamp"

        return self.app.evaluate_prediction(
            predictions_file,
            ground_truth_file,
            timestamp,
            name,
            model_id,
            display_it=display,
        )

    def dispatch(self, function_name: str, args: dict):
        """Wywołuje funkcję na podstawie jej nazwy i argumentów.
        Logi są jednocześnie wyświetlane na stdout/stderr (dla użytkownika)
        i przechwytywane (dla modelu).
        """
        if function_name not in self.function_map:
            return f"Error: Function '{function_name}' not found."

        func = self.function_map[function_name]

        capture_buffer = io.StringIO()
        tee_stdout = Tee(sys.stdout, capture_buffer)
        tee_stderr = Tee(sys.stderr, capture_buffer)

        try:
            # Przekierowujemy stdout ORAZ stderr do naszego rozgałęziacza (Tee)
            with (
                contextlib.redirect_stdout(tee_stdout),
                contextlib.redirect_stderr(tee_stderr),
            ):
                result = func(**args)

            captured_output = capture_buffer.getvalue()

            # Budujemy ostateczną odpowiedź dla modelu
            final_output = ""
            if captured_output:
                final_output += f"Logs and function output:\n{captured_output}\n"

            if result is not None:
                final_output += f"\nFunction returned result:\n{result}"

            if not final_output:
                final_output = "Function executed successfully (no output)."

            return final_output

        except Exception as e:
            return f"Error executing function '{function_name}': {e}"
