# Faza 1: Analiza i Projekt Architektury

Data: 2025-12-02

## 1. Identyfikacja Punktów Wejścia (Interfejsów Aplikacji)

Przeanalizowano plik `mlops/app.py` w celu zidentyfikowania publicznych metod klasy `Application`, które będą udostępnione w interfejsie czatu.

| Metoda                   | Opis                                                                                   | Parametry                                                                                                                                                            |
| ------------------------ | -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `generate_splits`        | Generuje kompletny zestaw podziałów danych dla eksperymentu.                             | `experiment_id`, `input_file`, `target_column`, `test_size`, `backtesting_strategy`, `cv_folds`, `backtesting_val_size`, `backtesting_test_size`, `perf_estimation_val_size`, `final_model_val_size` |
| `run_backtesting`        | Uruchamia proces backtestingu dla danego modelu i eksperymentu.                        | `experiment_id`, `model_config_name`                                                                                                                                 |
| `estimate_performance`   | Uruchamia proces estymacji wydajności dla modelu.                                      | `experiment_id`, `model_config_name`                                                                                                                                 |
| `train_final_model`      | Trenuje finalną wersję modelu.                                                         | `experiment_id`, `model_config_name`, `model_output_dir`                                                                                                             |
| `predict_new`            | Generuje predykcje dla nowych danych przy użyciu wytrenowanego modelu.                 | `model_path`, `input_file`, `prediction_id`, `timestamp`, `name`                                                                                                     |
| `evaluate_prediction`    | Ocenia predykcje w oparciu o dane referencyjne (ground truth).                           | `predictions_file`, `ground_truth_file`, `timestamp`, `name`, `model_id`, `display_it`                                                                               |
| `compare_models`         | Porównuje wyniki różnych modeli w ramach eksperymentu.                                 | `experiment_id`, `output_file`, `run_type`                                                                                                                           |
| `list_experiments`       | Wyświetla listę wszystkich dostępnych eksperymentów.                                   | Brak                                                                                                                                                                 |
| `list_models`            | Wyświetla listę wszystkich dostępnych konfiguracji modeli.                             | Brak                                                                                                                                                                 |

**Uwagi do implementacji:**

*   W metodach `predict_new` i `evaluate_prediction`, parametry takie jak `timestamp`, `prediction_id`, `model_id` powinny być generowane automatycznie przez aplikację czatu, a nie podawane przez użytkownika, aby uprościć interakcję.

## 2. Projekt Komunikacji z Gemini (Function Calling)

Aby umożliwić modelowi Gemini wywoływanie funkcji, zdefiniowano poniższe schematy (JSON Schema), które zostaną użyte w mechanizmie "Function Calling".

### Schematy Funkcji

#### `generate_splits`
```json
{
  "name": "generate_splits",
  "description": "Generuje kompletny zestaw podziałów danych (treningowe, walidacyjne, testowe) dla eksperymentu.",
  "parameters": {
    "type": "object",
    "properties": {
      "experiment_id": {
        "type": "string",
        "description": "Unikalny identyfikator eksperymentu, np. 'exp001'."
      },
      "input_file": {
        "type": "string",
        "description": "Ścieżka do wejściowego pliku CSV z danymi."
      },
      "target_column": {
        "type": "string",
        "description": "Nazwa kolumny docelowej (target)."
      },
      "test_size": {
        "type": "number",
        "description": "Procentowy udział zbioru testowego, np. 0.2."
      },
      "backtesting_strategy": {
        "type": "string",
        "enum": ["cv", "train-val"],
        "description": "Strategia podziału danych dla backtestingu."
      },
      "cv_folds": {
        "type": "integer",
        "description": "Liczba foldów dla strategii 'cv'."
      },
      "backtesting_val_size": {
        "type": "number",
        "description": "Rozmiar zbioru walidacyjnego dla strategii 'train-val'."
      },
      "backtesting_test_size": {
        "type": "number",
        "description": "Rozmiar zbioru testowego dla strategii 'train-val'."
      },
      "perf_estimation_val_size": {
        "type": "number",
        "description": "Rozmiar zbioru walidacyjnego dla estymacji wydajności."
      },
      "final_model_val_size": {
        "type": "number",
        "description": "Rozmiar zbioru walidacyjnego dla finalnego modelu."
      }
    },
    "required": ["experiment_id", "input_file", "target_column", "test_size", "backtesting_strategy"]
  }
}
```

#### `run_backtesting`
```json
{
  "name": "run_backtesting",
  "description": "Uruchamia proces backtestingu dla określonego modelu w ramach eksperymentu.",
  "parameters": {
    "type": "object",
    "properties": {
      "experiment_id": {
        "type": "string",
        "description": "Identyfikator eksperymentu, w którym ma być uruchomiony backtesting."
      },
      "model_config_name": {
        "type": "string",
        "description": "Nazwa konfiguracji modelu do użycia, np. 'roberta-base'."
      }
    },
    "required": ["experiment_id", "model_config_name"]
  }
}
```

#### `train_final_model`
```json
{
  "name": "train_final_model",
  "description": "Trenuje finalną wersję modelu na podstawie eksperymentu i zapisuje go.",
  "parameters": {
    "type": "object",
    "properties": {
      "experiment_id": {
        "type": "string",
        "description": "Identyfikator eksperymentu, na podstawie którego trenowany jest model."
      },
      "model_config_name": {
        "type": "string",
        "description": "Nazwa konfiguracji modelu do użycia."
      },
      "model_output_dir": {
        "type": "string",
        "description": "Katalog, w którym zostanie zapisany wytrenowany model."
      }
    },
    "required": ["experiment_id", "model_config_name", "model_output_dir"]
  }
}
```

#### `predict_new`
```json
{
  "name": "predict_new",
  "description": "Używa wytrenowanego modelu do generowania predykcji na nowych danych.",
  "parameters": {
    "type": "object",
    "properties": {
      "model_path": {
        "type": "string",
        "description": "Ścieżka do zapisanego modelu, który ma być użyty do predykcji."
      },
      "input_file": {
        "type": "string",
        "description": "Ścieżka do pliku CSV z nowymi danymi."
      },
      "name": {
        "type": "string",
        "description": "Nazwa dla tego zadania predykcji, np. 'predykcja_grudniowa'."
      }
    },
    "required": ["model_path", "input_file", "name"]
  }
}
```

#### `list_experiments`
```json
{
  "name": "list_experiments",
  "description": "Wyświetla listę wszystkich dostępnych eksperymentów.",
  "parameters": {
    "type": "object",
    "properties": {},
    "required": []
  }
}
```

#### `list_models`
```json
{
  "name": "list_models",
  "description": "Wyświetla listę wszystkich dostępnych modeli skonfigurowanych w systemie.",
  "parameters": {
    "type": "object",
    "properties": {},
    "required": []
  }
}
```
*Uwaga: Pozostałe funkcje (`estimate_performance`, `evaluate_prediction`, `compare_models`) będą miały analogiczne, proste schematy.*

## 3. Wybór Narzędzi

Zgodnie z planem, do implementacji zostaną użyte następujące biblioteki:

1.  **`google-generativeai`**: Do komunikacji z API Gemini i wykorzystania modelu językowego do NLU.
2.  **`rich`**: Do stworzenia estetycznego i interaktywnego interfejsu użytkownika w terminalu (TUI), w tym do kolorowania składni, wyświetlania tabel, pasków postępu i obsługi promptów.
