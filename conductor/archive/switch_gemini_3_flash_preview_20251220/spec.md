# Specification: Switch ChatML to gemini-3-flash-preview

## Overview
This track involves updating the ChatMLOps interface to use the `gemini-3-flash-preview` model by default. The model configuration will be centralized in `config.py` to improve maintainability and follow project patterns.

## Functional Requirements
1.  **Centralized Configuration:** Add `DEFAULT_CHAT_MODEL = "gemini-3-flash-preview"` to `config.py`.
2.  **Config-Driven Model Selection:** Update `GeminiConnector.get_model()` in `chatml/gemini_connector.py` to attempt to use `DEFAULT_CHAT_MODEL` first.
3.  **Graceful Fallback:** If the configured `DEFAULT_CHAT_MODEL` is not found in the list of available models from the Gemini API:
    -   Display a warning to the console using `rich`.
    -   Fall back to the current dynamic selection logic (first available model with "flash" in its name).
4.  **Logging:** Ensure the actually used model name is clearly printed to the console (already implemented, but should be verified).

## Non-Functional Requirements
-   Maintain compatibility with the existing `genai` library usage.
-   Adhere to the existing code style and dependency management (`uv`).

## Acceptance Criteria
-   ChatML starts successfully.
-   If `gemini-3-flash-preview` is available, it is selected and used.
-   If `gemini-3-flash-preview` is NOT available, a warning is shown, and another "flash" model is selected.
-   Model name is configurable in a single location (`config.py`).

## Out of Scope
-   Changing the model configuration for non-ChatML components (e.g., training pipelines) unless they already use a similar centralized config.
-   Implementing a CLI flag to override the model name (can be a future track).
