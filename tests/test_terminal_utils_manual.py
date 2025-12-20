import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.append(os.getcwd())

from mlops.terminal_utils import display_image

class TestTerminalUtils(unittest.TestCase):
    def test_display_image_no_support(self):
        with patch('mlops.terminal_utils.is_graphics_supported', return_value=False):
            with patch('sys.__stdout__') as mock_stdout:
                display_image("dummy_path")
                # Check if any call args contain the message
                found = False
                for call in mock_stdout.write.call_args_list:
                    if "Terminal does not support inline image display." in call[0][0]:
                        found = True
                        break
                self.assertTrue(found, "Did not find expected message in sys.__stdout__.write calls")

    def test_display_image_support(self):
        # Mock AutoImage to avoid actual file reading
        with patch('mlops.terminal_utils.is_graphics_supported', return_value=True):
            # Mock PIL Image.open to avoid file not found error
            with patch('mlops.terminal_utils.Image.open') as mock_image_open:
                mock_image_open.return_value = MagicMock() # Return a dummy image object
                
                with patch('mlops.terminal_utils.AutoImage') as mock_auto_image:
                    mock_img_instance = MagicMock()
                    # Configure __format__ to return a string
                    mock_img_instance.__format__ = MagicMock(return_value="IMAGE_DATA")
                    
                    mock_auto_image.return_value = mock_img_instance
                    
                    with patch('sys.__stdout__') as mock_stdout:
                        display_image("dummy_path")
                        
                        # We verify that we tried to write *something* derived from the image
                        mock_stdout.write.assert_called()
                        # Check that Image.open was called with the path
                        mock_image_open.assert_called_with("dummy_path")

if __name__ == "__main__":
    unittest.main()