# Terminal Image Display Implementation Plan

## Overview
The goal is to enable the ChatML CLI to display generated images (specifically model comparison plots) directly in the terminal, in addition to saving them to a file. The system must verify if the terminal supports image rendering and, if not, explicitly state that it cannot display the image. The file path must always be provided.

## Current State Analysis
-   **Image Generation**: `mlops/visualization.py` generates plots using `matplotlib` (Agg backend) and saves them as PNGs.
-   **Execution Flow**: `chatml/dispatcher.py` executes `mlops/app.py` methods using a `Tee` class that captures `stdout`/`stderr` to send back to the Gemini model.
-   **Constraint**: Printing raw image binary/escape codes to `stdout` will corrupt the context window of the Gemini model because `Tee` captures everything written to `sys.stdout`.
-   **Terminal Support**: Not all terminals support inline images. Common protocols include iTerm2, Kitty, and Sixel.

## Desired End State
-   When `visualize_comparison` (or similar) is called, the image is saved to disk.
-   The file path is printed to `stdout` (captured for Model & User).
-   The system checks if the terminal supports graphical image display (e.g., iTerm2, Kitty).
-   **If supported**: The image is rendered directly in the terminal using `sys.__stdout__` (bypassing the Model capture).
-   **If not supported**: A message "Terminal does not support inline image display." is printed to `sys.__stdout__` (or `stdout`).

### Key Discoveries
-   `chatml/dispatcher.py`: `Tee` class captures `stdout`. We must write to `sys.__stdout__` to show images to the user without sending them to the model.
-   `mlops/app.py`: `visualize_comparison` is the central point where the image path is known and execution happens.

## Implementation Approach
We will introduce a helper module `mlops/terminal_utils.py` to handle terminal capability detection and image rendering. We will utilize the `term-image` library (or a lightweight custom implementation if dependencies are restricted, but `term-image` is robust) to handle protocol details. To ensure we strictly follow "if not supported, say so" (avoiding ASCII fallbacks), we will configure the rendering to fail or check capabilities explicitly.

## Phase 1: Dependencies & Setup
### Overview
Add necessary libraries to handle terminal image rendering.

### Changes Required

#### 1. `pyproject.toml`
**File**: `pyproject.toml`
**Description**: Add `term-image` to dependencies.
```toml
dependencies = [
    # ... existing ...
    "term-image>=0.7.0",
]
```

### Success Criteria
-   [x] `uv sync` (or installation command) runs successfully.
-   [x] `import term_image` works in python shell.

---

## Phase 2: Terminal Utility Module
### Overview
Create a utility to handle the logic of checking support and printing images to the real stdout.

### Changes Required

#### 1. `mlops/terminal_utils.py` (New File)
**File**: `mlops/terminal_utils.py`
**Description**: Implement `display_image` function.
```python
import sys
import os
from term_image.image import AutoImage
from term_image.context import terminal_size

def is_graphics_supported():
    # Check specific env vars or use term-image's detection if reliable enough.
    # For strict adherence to "say if not supported", we might check:
    term_program = os.environ.get("TERM_PROGRAM", "")
    term = os.environ.get("TERM", "")
    
    # Common terminals with graphics support
    if "iTerm.app" in term_program: return True
    if "vscode" in term_program: return True
    if "wezterm" in term_program.lower(): return True
    if "kitty" in term: return True
    # Add more rigorous checks or trust term-image to fail gracefully if forced.
    return False

def display_image(image_path: str):
    """
    Displays image in terminal if supported, bypassing stdout capture.
    """
    if not is_graphics_supported():
        # Write to original stdout so user sees it, but maybe not model? 
        # actually, this info is useful for user, irrelevant for model.
        sys.__stdout__.write("Terminal does not support inline image display.\n")
        return

    try:
        # Create image object
        img = AutoImage(image_path)
        # Render to string
        # Note: AutoImage automatically selects best style. 
        # We want to ensure it's not just ASCII blocks unless that's acceptable.
        # The user implies "show image", usually meaning pixels.
        # term-image supports 'iterm2', 'kitty', 'sixel'.
        
        # Write directly to sys.__stdout__ to bypass Tee capture
        sys.__stdout__.write(f"{img:1.1}\n") # 1.1 forces execution/string conversion
        sys.__stdout__.flush()
    except Exception as e:
        sys.__stdout__.write(f"Failed to display image: {e}\n")

```

### Success Criteria
-   [x] Unit test or manual script to verify `display_image` writes to `__stdout__`.

---

## Phase 3: Integration
### Overview
Integrate the display logic into the MLOps application.

### Changes Required

#### 1. `mlops/app.py`
**File**: `mlops/app.py`
**Description**: Call `display_image` in `visualize_comparison` after saving the file.

```python
from mlops.terminal_utils import display_image

# ... inside visualize_comparison ...
# after plot_model_comparison call:
display_image(str(output_path))

return f"Wykres został wygenerowany i zapisany w: {output_path}"
```

### Success Criteria
-   [x] Running `visualize_comparison` prints the path AND the image (if in iTerm/VSCode).
-   [x] The captured logs (returned string) do NOT contain the image binary.

---

## Testing Strategy
### Manual Verification
1.  [x] **VSCode/iTerm2**: Run a command that triggers `visualize_comparison`. Verify image appears.
2.  [x] **Standard Terminal**: Run same command. Verify "Terminal does not support..." message.
3.  [x] **Log Check**: Inspect the "Logs and function output" returned to the model (debug print or check chat history) to ensure no massive base64 strings are present.

## Rollback Strategy
-   Remove `term-image` from `pyproject.toml`.
-   Delete `mlops/terminal_utils.py`.
-   Revert changes to `mlops/app.py`.
