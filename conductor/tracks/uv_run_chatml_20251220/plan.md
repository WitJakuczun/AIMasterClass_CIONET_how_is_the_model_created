# Track Plan: Enable `uv run chatml` Command

## Phase 1: Infrastructure and Entry Point [checkpoint: 7509159]
This phase focuses on updating the project metadata and creating the necessary Python wrapper to support the new command.

- [x] Task: Create synchronous wrapper `run_chat` in `chatml/chat.py` c2c132c
    - **Description:** Implement a synchronous function that initializes the asyncio loop and runs the main chat function.
    - **Write Tests:** Create `tests/test_chat_entry.py` to verify `run_chat` can be imported and correctly calls the underlying main logic (using mocks).
    - **Implement:** Add `run_chat()` to `chatml/chat.py`.
    - **Verification:** Run `pytest tests/test_chat_entry.py`.

- [x] Task: Update `pyproject.toml` with `chatml` script d09ee73
    - **Description:** Add the `chatml` entry point to the `[project.scripts]` section.
    - **Write Tests:** Add a test case to `tests/test_chat_entry.py` that verifies the entry point string is correctly formatted.
    - **Implement:** Edit `pyproject.toml`.
    - **Verification:** Run `uv sync` and then `uv run chatml --help` (or equivalent) to ensure the command is recognized.

- [x] Task: Conductor - User Manual Verification 'Infrastructure and Entry Point' (Protocol in workflow.md)

## Phase 2: Final Verification and Cleanup
This phase ensures the new command works end-to-end and the environment is synchronized.

- [ ] Task: End-to-End Manual Verification
    - **Description:** Manually launch the chat using the new command and perform a simple interaction.
    - **Verification:** Execute `uv run chatml`, type "List all available models", and verify the response.

- [ ] Task: Conductor - User Manual Verification 'Final Verification and Cleanup' (Protocol in workflow.md)
