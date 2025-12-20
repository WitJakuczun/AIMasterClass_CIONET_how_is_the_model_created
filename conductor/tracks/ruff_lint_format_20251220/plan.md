# Plan: Use Ruff for Linter and Code Formatting

## Phase 1: Environment Setup & Configuration [checkpoint: aeb5d24]
- [x] Task: Add `ruff` to project development dependencies using `uv`. f444b29
- [x] Task: Define Ruff configuration in `pyproject.toml` (lint rules: E, F, I, D, S; format settings). f5f6d7f
- [x] Task: Create a verification test that ensures Ruff configuration is valid and reachable. 75767c0
- [x] Task: Conductor - User Manual Verification 'Environment Setup & Configuration' (Protocol in workflow.md)

## Phase 2: Initial Codebase Cleanup
- [ ] Task: Execute `uv run ruff format .` to apply consistent formatting to the entire repository.
- [ ] Task: Execute `uv run ruff check --fix .` to resolve auto-fixable linting issues (imports, basic rules).
- [ ] Task: Manually resolve any remaining critical linting/security issues identified by Ruff.
- [ ] Task: Conductor - User Manual Verification 'Initial Codebase Cleanup' (Protocol in workflow.md)

## Phase 3: Automation & Integration
- [ ] Task: Create or update `.pre-commit-config.yaml` to include Ruff as a pre-commit hook.
- [ ] Task: Verify pre-commit integration by staging a change and running `pre-commit run`.
- [ ] Task: Update `README.md` or documentation with instructions for using Ruff commands.
- [ ] Task: Conductor - User Manual Verification 'Automation & Integration' (Protocol in workflow.md)
