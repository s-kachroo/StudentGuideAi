import unittest
from ui import app
from prompt_engine import get_chat_response

class ChatbotTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_index_page(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Chat with Bot', response.data)

    def test_chat_endpoint(self):
        response = self.app.post('/chat', data=dict(user_input='Hello'))
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'response', response.data)

    def test_prompt_engine(self):
        response = get_chat_response("Hello")
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

if __name__ == '__main__':
    unittest.main()
