# chatml/function_schemas.py

# Definicje narzędzi (funkcji), które model Gemini może wywołać
# Na podstawie analizy z pliku: 20251202-chatml-analiza_projekt.md

tools = {
    "generate_splits": {
        "name": "generate_splits",
        "description": "Generuje kompletny zestaw podziałów danych (treningowe, walidacyjne, testowe) dla eksperymentu.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "experiment_id": {"type": "STRING", "description": "Unikalny identyfikator eksperymentu, np. 'exp001'."},
                "input_file": {"type": "STRING", "description": "Ścieżka do wejściowego pliku CSV z danymi."},
                "target_column": {"type": "STRING", "description": "Nazwa kolumny docelowej (target)."},
                "test_size": {"type": "NUMBER", "description": "Procentowy udział zbioru testowego, np. 0.2."},
                "backtesting_strategy": {"type": "STRING", "enum": ["cv", "train-val"], "description": "Strategia podziału danych dla backtestingu."},
                "cv_folds": {"type": "INTEGER", "description": "Liczba foldów dla strategii 'cv'."},
                "backtesting_val_size": {"type": "NUMBER", "description": "Rozmiar zbioru walidacyjnego dla strategii 'train-val'."},
                "backtesting_test_size": {"type": "NUMBER", "description": "Rozmiar zbioru testowego dla strategii 'train-val'."},
                "perf_estimation_val_size": {"type": "NUMBER", "description": "Rozmiar zbioru walidacyjnego dla estymacji wydajności."},
                "final_model_val_size": {"type": "NUMBER", "description": "Rozmiar zbioru walidacyjnego dla finalnego modelu."}
            },
            "required": ["experiment_id", "input_file", "target_column", "test_size", "backtesting_strategy"]
        }
    },
    "run_backtesting": {
        "name": "run_backtesting",
        "description": "Uruchamia proces backtestingu dla określonego modelu w ramach eksperymentu.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "experiment_id": {"type": "STRING", "description": "Identyfikator eksperymentu, w którym ma być uruchomiony backtesting."},
                "model_config_name": {"type": "STRING", "description": "Nazwa konfiguracji modelu do użycia, np. 'roberta-base'."}
            },
            "required": ["experiment_id", "model_config_name"]
        }
    },
    "estimate_performance": {
        "name": "estimate_performance",
        "description": "Uruchamia proces estymacji wydajności dla modelu.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "experiment_id": {"type": "STRING", "description": "Identyfikator eksperymentu."},
                "model_config_name": {"type": "STRING", "description": "Nazwa konfiguracji modelu do użycia."}
            },
            "required": ["experiment_id", "model_config_name"]
        }
    },
    "train_final_model": {
        "name": "train_final_model",
        "description": "Trenuje finalną wersję modelu na podstawie eksperymentu i zapisuje go.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "experiment_id": {"type": "STRING", "description": "Identyfikator eksperymentu, na podstawie którego trenowany jest model."},
                "model_config_name": {"type": "STRING", "description": "Nazwa konfiguracji modelu do użycia."},
                "model_output_dir": {"type": "STRING", "description": "Katalog, w którym zostanie zapisany wytrenowany model."}
            },
            "required": ["experiment_id", "model_config_name", "model_output_dir"]
        }
    },
    "predict_new": {
        "name": "predict_new",
        "description": "Używa wytrenowanego modelu do generowania predykcji na nowych danych.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "model_path": {"type": "STRING", "description": "Ścieżka do zapisanego modelu, który ma być użyty do predykcji."},
                "input_file": {"type": "STRING", "description": "Ścieżka do pliku CSV z nowymi danymi."},
                "name": {"type": "STRING", "description": "Nazwa dla tego zadania predykcji, np. 'predykcja_grudniowa'."}
            },
            "required": ["model_path", "input_file", "name"]
        }
    },
    "evaluate_prediction": {
        "name": "evaluate_prediction",
        "description": "Ocenia jakość predykcji modelu, porównując je z danymi referencyjnymi (ground truth).",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "predictions_file": {"type": "STRING", "description": "Ścieżka do pliku CSV z predykcjami."},
                "ground_truth_file": {"type": "STRING", "description": "Ścieżka do pliku CSV z prawdziwymi etykietami (ground truth)."},
                "display": {"type": "BOOLEAN", "description": "Czy wyświetlić wyniki w konsoli (domyślnie True)."}
            },
            "required": ["predictions_file", "ground_truth_file"]
        }
    },
    "list_experiments": {
        "name": "list_experiments",
        "description": "Wyświetla listę wszystkich dostępnych eksperymentów.",
        "parameters": {}
    },
    "list_models": {
        "name": "list_models",
        "description": "Wyświetla listę wszystkich dostępnych modeli skonfigurowanych w systemie.",
        "parameters": {}
    },
    "visualize_comparison": {
        "name": "visualize_comparison",
        "description": "Generates a bar chart comparing model metrics (e.g., f1_macro, accuracy) for a given experiment and saves it as an image file. Useful for visual analysis of model performance.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "experiment_id": {
                    "type": "STRING",
                    "description": "The unique identifier of the experiment (e.g., 'exp001')."
                },
                "metric": {
                    "type": "STRING",
                    "description": "The metric to compare models on. Defaults to 'f1_macro'. Options: 'accuracy', 'precision_macro', 'recall_macro', 'f1_macro'."
                },
                "run_type": {
                    "type": "STRING",
                    "description": "The stage of the experiment. Defaults to 'backtesting'. Options: 'backtesting', 'performance_estimation'."
                }
            },
            "required": ["experiment_id"]
        }
    },
    "compare_models": {
        "name": "compare_models",
        "description": "Porównuje wyniki różnych modeli w ramach eksperymentu.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "experiment_id": {"type": "STRING", "description": "Identyfikator eksperymentu do porównania."},
                "output_file": {"type": "STRING", "description": "Opcjonalna ścieżka do pliku wyjściowego dla raportu."},
                "run_type": {"type": "STRING", "enum": ["backtesting", "performance_estimation"], "description": "Typ uruchomienia do porównania (opcjonalnie)."}
            },
            "required": ["experiment_id"]
        }
    }
}