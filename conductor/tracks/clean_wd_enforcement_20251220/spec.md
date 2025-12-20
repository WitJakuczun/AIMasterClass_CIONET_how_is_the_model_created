# Specification: Clean Working Directory Enforcement & Stashing

## Overview
This track introduces a mandatory check for uncommitted changes during the track initialization (branching) process. This ensures that work is not accidentally carried across branches and promotes a clean development workflow.

## Functional Requirements
1.  **Guiding Principle Update:** Add a new principle to `conductor/workflow.md` regarding the maintenance of a clean working directory before branching.
2.  **Pre-Branching Check:** Implement a mandatory check for uncommitted files as the first step of the **Track/Feature Initialization (Branching)** procedure.
3.  **Interactive Handling of Changes:** If uncommitted changes are detected, present the user with an interactive menu:
    -   **Option 1: Stash changes:** Promptly stash the changes with a descriptive message (e.g., "Auto-stash before branching to [new_branch_name]").
    -   **Option 2: Cancel:** Abort the branching process to allow manual handling.
    -   **Option 3: Proceed Anyway:** (Included with a strong warning about potential conflicts).
4.  **Post-Branching Stash Management:** If changes were stashed (Option 1):
    -   After creating and switching to the new branch, ask the user: "Would you like to apply the stashed changes to this new branch now, or keep them stashed?"
    -   Handle the `git stash pop` or leave the stash as-is based on the response.

## Non-Functional Requirements
-   **Clarity:** Use `rich` formatting to make detected changes and menu options visually clear.
-   **Safety:** Ensure the stashing process is robust and provides clear feedback on success or failure.

## Acceptance Criteria
-   The "Guiding Principles" in `workflow.md` are updated.
-   Starting a new track with uncommitted changes triggers the interactive menu.
-   Stashing, switching branches, and optionally popping the stash works as specified.
-   The process can be cancelled without losing work.

## Out of Scope
-   Automated merging of stashes across multiple branches.
-   A complex GUI for stashing specific files (all-or-nothing approach for simplicity).
