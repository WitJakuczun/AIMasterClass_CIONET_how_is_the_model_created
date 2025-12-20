import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

class TestGeminiSchema(unittest.TestCase):
    @patch('chatml.gemini_connector.genai')
    def test_tools_schema_no_default(self, mock_genai):
        from chatml.gemini_connector import GeminiConnector
        
        # Mock list_models to return a valid model
        mock_model = MagicMock()
        mock_model.name = 'gemini-1.5-flash'
        mock_model.supported_generation_methods = ['generateContent']
        mock_genai.list_models.return_value = iter([mock_model])
        
        connector = GeminiConnector()
        connector.model = None # Ensure clean state
        
        # Call get_model which should initialize GenerativeModel with tools
        connector.get_model()
        
        # Verify GenerativeModel was called
        self.assertTrue(mock_genai.GenerativeModel.called)
        
        # Get arguments passed to GenerativeModel
        args, kwargs = mock_genai.GenerativeModel.call_args
        
        # 'tools' should be in kwargs or args[1]
        tools = kwargs.get('tools')
        if not tools and len(args) > 1:
            tools = args[1]
            
        self.assertIsNotNone(tools, "Tools were not passed to GenerativeModel")
        
        # Iterate through all tools and parameters to check for 'default' key
        for tool in tools:
             params = tool.get('parameters', {}).get('properties', {})
             for param_name, param_def in params.items():
                 self.assertNotIn('default', param_def, f"Parameter '{param_name}' in tool '{tool['name']}' has a forbidden 'default' key.")

if __name__ == '__main__':
    unittest.main()
