import sys
import os
from term_image.image import AutoImage
from PIL import Image

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
        # term-image requires a PIL Image instance, not a path string
        pil_img = Image.open(image_path)
        img = AutoImage(pil_img)
        
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
