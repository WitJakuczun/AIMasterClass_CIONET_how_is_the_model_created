# Plan: Switch ChatML to gemini-3-flash-preview

## Phase 1: Configuration Setup [checkpoint: fa086f1]
- [x] Task: Update `config.py` with `DEFAULT_CHAT_MODEL = "gemini-3-flash-preview"` e4586c2
- [ ] Task: Conductor - User Manual Verification 'Configuration Setup' (Protocol in workflow.md)

## Phase 2: Connector Logic (TDD)
- [x] Task: Create `tests/test_gemini_connector.py` and write failing tests for `GeminiConnector.get_model()` (Red Phase) e6488d0
    - Sub-task: Test successful selection when `DEFAULT_CHAT_MODEL` is available.
    - Sub-task: Test fallback and warning when `DEFAULT_CHAT_MODEL` is NOT available.
- [x] Task: Implement config-driven model selection and fallback in `chatml/gemini_connector.py` (Green Phase) 1b1e967
- [x] Task: Verify tests pass and check coverage 38578a3
- [ ] Task: Conductor - User Manual Verification 'Connector Logic' (Protocol in workflow.md)

## Phase 3: Final Integration & Quality Gate
- [ ] Task: Run full test suite (`pytest`) and linting (`ruff`)
- [ ] Task: Conductor - User Manual Verification 'Final Integration & Quality Gate' (Protocol in workflow.md)
