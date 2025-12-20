# Track Specification: Enable `uv run chatml` Command

## Overview
Currently, the ChatMLOps interface is launched using `uv run python -m chatml.chat`. This track aims to simplify the developer experience by providing a direct command shortcut, `chatml`, allowing the interface to be started via `uv run chatml`.

## Functional Requirements
1.  **Script Registration:** Add a `chatml` entry point to the `[project.scripts]` section of `pyproject.toml`.
2.  **Entry Point Implementation:** Create a synchronous wrapper function `run_chat()` in `chatml/chat.py`.
    -   This function will act as the designated entry point for the script.
    -   It will encapsulate the `asyncio.run(main())` call to properly initialize the asynchronous event loop.
3.  **Environment Sync:** Ensure `uv.lock` is updated to reflect the changes in `pyproject.toml`.

## Technical Details
-   **Target File:** `pyproject.toml`
-   **New Entry:** `chatml = "chatml.chat:run_chat"`
-   **Target File:** `chatml/chat.py`
-   **New Function:** 
    ```python
    def run_chat():
        asyncio.run(main())
    ```

## Acceptance Criteria
-   Executing `uv run chatml` successfully launches the ChatMLOps interactive session.
-   The existing method `uv run python -m chatml.chat` continues to function correctly.
-   All existing functionality within the chat interface remains unaffected.

## Out of Scope
-   Adding additional command aliases or shortcuts.
-   Refactoring the internal logic of the chat interface beyond creating the wrapper.
