# Track Specification: Migrate to Polars and Rich

## Goal
The goal of this track is to modernize the application's data processing and user interface layers. This involves replacing the `pandas` library with `polars` for faster and more memory-efficient data manipulation, and integrating the `rich` library to provide a visually engaging and professional CLI experience during the presentation.

## User Stories
- **As a Data Scientist (User):** I want the data processing steps (loading, splitting, manipulating) to be noticeably faster, especially when generating large synthetic datasets.
- **As a Presenter:** I want the CLI output to be colorful, structured, and interactive (e.g., progress bars, tables) so that the audience stays engaged and can easily follow the demo.
- **As a Developer:** I want to use modern, efficient libraries (`polars`, `rich`) to keep the codebase maintainable and performant.

## Functional Requirements

### 1. Data Processing Migration (Pandas -> Polars)
- **Dependency:** Remove `pandas` (unless strictly required by `scikit-learn` or `transformers` internals where interoperability is needed) and add `polars`.
- **Data Loading:** Update `mlops.experiment_service` and `data.generator` to read CSV/Parquet files using `polars.read_csv`.
- **Data Manipulation:** Refactor `mlops.splitting` and `mlops.pipeline` to use Polars expressions (`filter`, `select`, `with_columns`) instead of Pandas DataFrames.
- **Data Saving:** Ensure all intermediate and final artifacts are saved using `polars.DataFrame.write_csv`.
- **Compatibility:** Ensure `mlops.dataset.Dataset` (Hugging Face) loading still works. This might require converting Polars DataFrames to Arrow or Pandas (if necessary) before passing to HF `Dataset.from_pandas` (or `from_polars` if supported).

### 2. CLI UI Enhancement (Rich)
- **Dependency:** Add `rich`.
- **Logging:** Configure `loguru` to use `rich` for formatting logs (colors, timestamps, levels).
- **Tables:** Replace any basic console prints of data summaries (e.g., in `mlops.visualization` or `compare-models` command) with `rich.table.Table`.
- **Progress Bars:** Implement `rich.progress` for long-running operations:
    - Data generation
    - Model training (if feasible to hook into the callback)
    - Evaluation loops
- **Status Spinners:** Use `rich.console.Console.status` for indeterminate waiting periods (e.g., "Loading model...", "Saving artifacts...").

## Non-Functional Requirements
- **Performance:** Data loading and transformation operations should be faster or at least on par with the Pandas implementation.
- **Visuals:** The terminal output must be clean, readable, and use the project's color scheme (if defined) or standard sensible defaults (green for success, red for error).
- **Testing:** All existing tests (especially `tests/test_experiment.py`) must pass after the migration. New tests should be added if logic changes significantly.

## Technical Constraints
- `scikit-learn` often expects Numpy arrays or Pandas DataFrames. Ensure appropriate conversion (`to_numpy()` or `to_pandas()`) is done only at the boundaries where these libraries interact.
- Hugging Face `datasets` library interacts well with Arrow. Polars is built on Arrow, so leverage zero-copy conversions where possible.
