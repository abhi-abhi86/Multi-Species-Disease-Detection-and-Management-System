import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the project root to the Python path to allow for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.llm_integrator import LLMIntegrator

class TestLLMIntegratorCache(unittest.TestCase):

    def setUp(self):
        """Set up a new LLMIntegrator instance for each test."""
        # We pass a dummy API key to enable the LLM features for testing
        self.integrator = LLMIntegrator(api_key="test_key")

    @patch('core.llm_integrator.openai.OpenAI')
    def test_cache_hit(self, MockOpenAI):
        """Test that a repeated query hits the cache and doesn't call the API again."""
        # Arrange: Mock the OpenAI client and its response
        mock_response = MagicMock()
        mock_response.choices[0].message = {'content': 'This is a test response.'}
        
        mock_client_instance = MockOpenAI.return_value
        mock_client_instance.chat.completions.create.return_value = mock_response

        query = "What is Potato Early Blight?"

        # Act: Call the function twice with the same query
        first_response = self.integrator.generate_response(query)
        second_response = self.integrator.generate_response(query)

        # Assert
        # The API should only be called once
        mock_client_instance.chat.completions.create.assert_called_once()
        self.assertEqual(first_response, "This is a test response.")
        self.assertEqual(second_response, "This is a test response.")
        self.assertEqual(len(self.integrator.cache), 1)

    @patch('core.llm_integrator.openai.OpenAI')
    def test_cache_miss(self, MockOpenAI):
        """Test that different queries miss the cache and call the API each time."""
        # Arrange
        mock_client_instance = MockOpenAI.return_value
        mock_client_instance.chat.completions.create.return_value = MagicMock()

        # Act
        self.integrator.generate_response("Query 1")
        self.integrator.generate_response("Query 2")

        # Assert
        # The API should be called twice for two different queries
        self.assertEqual(mock_client_instance.chat.completions.create.call_count, 2)
        self.assertEqual(len(self.integrator.cache), 2)

    @patch('core.llm_integrator.openai.OpenAI')
    def test_cache_eviction(self, MockOpenAI):
        """Test that the oldest cache item is evicted when the cache is full."""
        # Arrange
        mock_client_instance = MockOpenAI.return_value
        mock_client_instance.chat.completions.create.side_effect = lambda **kwargs: MagicMock(choices=[MagicMock(message={'content': f"Response to {kwargs['messages'][-1]['content']}"})])

        # The default cache size is 50. We'll add 50 items.
        for i in range(50):
            self.integrator.generate_response(f"Query {i}")

        # Assert: Cache is full
        self.assertEqual(len(self.integrator.cache), 50)
        self.assertIn(("Query 0", None), self.integrator.cache)

        # Act: Add one more item to trigger eviction
        self.integrator.generate_response("Query 50")

        # Assert: The oldest item ("Query 0") should be gone
        self.assertEqual(len(self.integrator.cache), 50)
        self.assertNotIn(("Query 0", None), self.integrator.cache)
        self.assertIn(("Query 50", None), self.integrator.cache)

if __name__ == '__main__':
    unittest.main()