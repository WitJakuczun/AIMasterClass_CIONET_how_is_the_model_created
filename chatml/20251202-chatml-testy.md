# Plan Testów dla Aplikacji ChatMLOps

Data: 2025-12-02

## Cel

Celem jest zweryfikowanie poprawności działania kluczowych funkcjonalności aplikacji `chatml`, w tym uruchamiania, uwierzytelniania, przetwarzania poleceń w języku naturalnym, wywoływania funkcji MLOps oraz obsługi błędów.

---

## 1. Testy Uruchomienia i Interfejsu

| ID Testu | Opis Akcji | Oczekiwany Rezultat |
| :--- | :--- | :--- |
| **T1.01** | Uruchom aplikację poleceniem `uv run python -m chatml.chat`. | Aplikacja startuje, wyświetla panel powitalny i prosi o wprowadzenie polecenia. |
| **T1.02** | Wpisz `exit` i naciśnij Enter. | Aplikacja wyświetla panel pożegnalny i kończy działanie. |
| **T1.03** | W trakcie działania aplikacji, naciśnij `Ctrl+C`. | Aplikacja przechwytuje sygnał, wyświetla panel pożegnalny i kończy działanie. |

## 2. Testy Uwierzytelniania

| ID Testu | Warunki Wstępne | Opis Akcji | Oczekiwany Rezultat |
| :--- | :--- | :--- | :--- |
| **T2.01** | Skonfigurowane poświadczenia ADC (`gcloud auth application-default login`), zmienna `GEMINI_API_KEY` **nie** jest ustawiona. | Uruchom aplikację. | Aplikacja wyświetla komunikat o pomyślnym uwierzytelnieniu przez ADC i działa normalnie. |
| **T2.02** | Poświadczenia ADC **nie** są skonfigurowane, zmienna `GEMINI_API_KEY` jest ustawiona z poprawnym kluczem. | Uruchom aplikację. | Aplikacja wyświetla komunikat o próbie użycia ADC, następnie o użyciu klucza API i działa normalnie. |
| **T2.03** | Poświadczenia ADC **nie** są skonfigurowane, zmienna `GEMINI_API_KEY` **nie** jest ustawiona. | Uruchom aplikację. | Aplikacja wyświetla komunikat o błędzie braku poświadczeń i kończy działanie. |
| **T2.04** | Poświadczenia ADC **nie** są skonfigurowane, zmienna `GEMINI_API_KEY` jest ustawiona z **niepoprawnym** kluczem. | Uruchom aplikację. | Aplikacja wyświetla komunikat o błędzie podczas konfiguracji przy użyciu klucza API i kończy działanie. |

## 3. Testy Przetwarzania Poleceń (NLU)

| ID Testu | Wprowadzone Polecenie | Oczekiwany Rezultat |
| :--- | :--- | :--- |
| **T3.01** | `pokaż dostępne modele` | Gemini poprawnie rozpoznaje intencję i zwraca wywołanie funkcji `list_models` bez argumentów. |
| **T3.02** | `chcę zobaczyć listę modeli` | Gemini poprawnie rozpoznaje intencję i zwraca wywołanie funkcji `list_models` bez argumentów. |
| **T3.03**| `uruchom backtesting dla modelu roberta-base w eksperymencie exp001` | Gemini poprawnie rozpoznaje intencję i zwraca wywołanie `run_backtesting` z argumentami: `model_config_name='roberta-base'`, `experiment_id='exp001'`. |
| **T3.04** | `przetestuj model svm-opt na eksperymencie exp002` | Gemini poprawnie rozpoznaje intencję i zwraca wywołanie `run_backtesting` z argumentami: `model_config_name='svm-opt'`, `experiment_id='exp002'`. |
| **T3.05** | `jaka jest pogoda?` (polecenie nie związane z funkcjami) | Gemini odpowiada w języku naturalnym, nie wywołując żadnej funkcji. Aplikacja wyświetla tę odpowiedź. |
| **T3.06** | `uruchom backtesting` (polecenie niekompletne) | Gemini powinno dopytać o brakujące parametry (np. "Dla jakiego modelu i eksperymentu?") lub zwrócić odpowiedź tekstową informującą o braku danych. Aplikacja powinna obsłużyć odpowiedź. |

## 4. Testy Wykonywania Funkcji (Dispatching)

| ID Testu | Wprowadzone Polecenie | Oczekiwany Rezultat |
| :--- | :--- | :--- |
| **T4.01** | `pokaż listę modeli` | Aplikacja wywołuje `dispatcher.dispatch('list_models', {})`. Przechwycone standardowe wyjście z funkcji (lista modeli) jest poprawnie wyświetlane w panelu wynikowym. |
| **T4.02** | `pokaż eksperymenty` | Aplikacja wywołuje `dispatcher.dispatch('list_experiments', {})`. Przechwycone standardowe wyjście z funkcji (lista eksperymentów) jest poprawnie wyświetlane w panelu wynikowym. |
| **T4.03** | Polecenie wymagające argumentów, np. `uruchom backtesting...` | Aplikacja poprawnie parsuje argumenty, wywołuje `dispatcher.dispatch` z odpowiednimi `args`. Logi z wykonania funkcji są przechwytywane i wyświetlane. |


## 5. Testy Obsługi Błędów

| ID Testu | Opis Akcji | Oczekiwany Rezultat |
| :--- | :--- | :--- |
| **T5.01** | Wprowadź polecenie, które wywoła funkcję z nieistniejącym plikiem, np. `run_backtesting` dla eksperymentu, który nie ma danych. | Aplikacja przechwytuje wyjątek z `dispatcher` i wyświetla w panelu czytelny komunikat o błędzie, np. "Błąd podczas wykonywania funkcji 'run_backtesting': FileNotFoundError(...)". |
| **T5.02** | Wprowadź polecenie z nieistniejącą nazwą modelu. | Aplikacja przechwytuje wyjątek i wyświetla komunikat o błędzie informujący o nieznalezionym modelu. |
| **T5.03** | Wprowadź losowy ciąg znaków, np. "asdasd dfgdfg". | Model Gemini powinien odpowiedzieć, że nie zrozumiał polecenia. Aplikacja powinna wyświetlić tę odpowiedź. |
