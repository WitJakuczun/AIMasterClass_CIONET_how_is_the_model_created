# Track Plan: Migrate to Polars and Rich

## Phase 1: Setup and Dependency Management [checkpoint: 5ce4b89]
This phase focuses on preparing the environment by swapping the core data handling dependencies and setting up the visual foundation.

- [x] Task: Update Dependencies (Pandas -> Polars, Add Rich) <!-- id: d2c1c3c -->
    - **Description:** Remove `pandas` from `pyproject.toml` (or move to dev-dependencies if strictly needed for tests) and add `polars` and `rich`. Update `uv.lock`.
    - **Technical Details:** Edit `pyproject.toml`. Run `uv sync` or `uv lock`.
    - **Verification:** Run `uv run python -c "import polars; import rich"` to confirm installation.

- [x] Task: Configure Rich Logging <!-- id: bf7a3ee -->
    - **Description:** Update the `loguru` configuration in the entry point (`main.py` or `mlops/__init__.py`) to use `rich.logging.RichHandler`.
    - **Technical Details:** Modify the `logger.add` call. Ensure logs are beautifully formatted with timestamps and colors.
    - **Verification:** Run `uv run cli list-models` and check the log output format.

- [ ] Task: Conductor - User Manual Verification 'Setup and Dependency Management' (Protocol in workflow.md)

## Phase 2: Data Layer Migration (Pandas -> Polars)
This phase involves the deep refactoring of the data processing logic. We will tackle this module by module to ensure stability.

- [x] Task: Refactor Data Generator (`data/generator.py`) <!-- id: 2a3a624 -->
    - **Description:** Update the data generation logic to use Polars for creating and manipulating the initial synthetic datasets.
    - **Technical Details:** Replace `pd.DataFrame` with `pl.DataFrame`. Use `pl.concat` instead of `pd.concat`. Update CSV writing.
    - **Verification:** Run `uv run cli data-generate --name test_gen --examples-per-sentiment 10` and verify the output CSV is valid.

- [x] Task: Refactor Data Splitting (`mlops/splitting.py`) <!-- id: 0baf11d -->
    - **Description:** Rewrite the `StratifiedSplitter` or equivalent logic to handle Polars DataFrames.
    - **Technical Details:** Polars does not have a direct `StratifiedShuffleSplit` equivalent like sklearn on pandas. You may need to:
        1. Convert to Pandas/Numpy just for the `sklearn` split call (boundary conversion).
        2. Or implement a native Polars stratified split (groupby + sample).
        *Decision:* Use boundary conversion for now to minimize risk, i.e., `df.to_pandas()` -> `sklearn` -> `df[train_idx]`.
    - **Verification:** Run `uv run cli generate-splits` and check if train/val/test splits are created correctly.

- [ ] Task: Refactor Experiment Service (`mlops/experiment_service.py`)
    - **Description:** Update how the service loads data for experiments. It should return Polars DataFrames.
    - **Technical Details:** Change `pd.read_csv` to `pl.read_csv`.
    - **Verification:** Run existing tests `tests/test_experiment.py`. They will likely fail; this task prepares the ground for fixing them.

- [ ] Task: Refactor Pipeline and Model Interface Boundaries (`mlops/pipeline.py`, `mlops/model_interface.py`)
    - **Description:** The Hugging Face `Dataset` object usually consumes the data. Ensure the transition from Polars to HF Dataset is smooth.
    - **Technical Details:** In `mlops/pipeline.py`, update the code that feeds the tokenizer. `Dataset.from_polars(df)` might be available or use `Dataset(pa.Table.from_pandas(df.to_arrow()))`.
    - **Verification:** Run a small backtesting run: `uv run cli run-backtesting ...` (will fail until all consumers are updated, but check for data loading success).

- [ ] Task: Conductor - User Manual Verification 'Data Layer Migration (Pandas -> Polars)' (Protocol in workflow.md)

## Phase 3: UI Enhancement (Rich)
Now that the data layer is efficient, we will make the user experience engaging.

- [ ] Task: Enhance `list-models` and `list-experiments` with Rich Tables
    - **Description:** Replace standard text printing with `rich.table.Table`.
    - **Technical Details:** In `mlops/app.py` or the specific command handlers, create a `Table` object, add columns, and add rows. Print via `Console`.
    - **Verification:** Run `uv run cli list-models` and observe the rendered table.

- [ ] Task: Add Progress Bars to Training and Evaluation
    - **Description:** Wrap long running loops (like the fold iteration in backtesting or the generation loop) with `rich.progress.track` or `Progress`.
    - **Technical Details:** In `mlops/runner.py`, locate the loops for folds/seeds. Add the progress bar context manager.
    - **Verification:** Run `uv run cli run-backtesting` and see the progress bar in action.

- [ ] Task: Add Status Spinners for Data Loading
    - **Description:** Show a "Loading data..." spinner during file I/O operations.
    - **Technical Details:** Use `with console.status("Loading..."):` blocks in `mlops/experiment_service.py` or the main command entry points.
    - **Verification:** Run any command requiring data loading and check for the spinner.

- [ ] Task: Conductor - User Manual Verification 'UI Enhancement (Rich)' (Protocol in workflow.md)

## Phase 4: Verification and Cleanup
Final checks to ensure everything works as expected.

- [ ] Task: Fix and Run All Tests
    - **Description:** Ensure that all unit tests (`tests/`) are compatible with the new Polars data structures.
    - **Technical Details:** Update test data fixtures to be Polars DataFrames if necessary. Run `pytest`.
    - **Verification:** All tests pass.

- [ ] Task: Final Manual Demo Run
    - **Description:** Execute the full "Happy Path" demo script to ensure the presentation flow is smooth.
    - **Technical Details:**
        1. `data-generate`
        2. `generate-splits`
        3. `run-backtesting`
        4. `compare-models` (check table output)
    - **Verification:** No crashes, beautiful output, fast execution.

- [ ] Task: Conductor - User Manual Verification 'Verification and Cleanup' (Protocol in workflow.md)
