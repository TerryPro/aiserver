import os
import shutil
import unittest
import sys

# Ensure aiserver is in path
sys.path.append(os.getcwd())

from langchain_core.messages import HumanMessage, AIMessage
from aiserver.core.session import SessionManager
from aiserver.core.history import JuChatMessageHistory

class TestJuChatMessageHistory(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for sessions
        self.test_dir = ".test_sessions"
        self.session_manager = SessionManager(storage_dir=self.test_dir)
        self.notebook_id = "nb-test"
        self.cell_id = "cell-1"
        self.current_code = "print('hello')"

    def tearDown(self):
        # Cleanup
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_add_messages_pair(self):
        history = JuChatMessageHistory(
            self.session_manager, 
            self.notebook_id, 
            self.cell_id, 
            self.current_code
        )

        messages = [
            HumanMessage(content="Hello AI"),
            AIMessage(content="Hello User")
        ]
        
        history.add_messages(messages)
        
        # Verify persistence
        stored_messages = history.messages
        self.assertEqual(len(stored_messages), 2)
        self.assertIsInstance(stored_messages[0], HumanMessage)
        # Note: SessionManager combines intent and code in the retrieved history
        self.assertIn("Intent: Hello AI", stored_messages[0].content)
        self.assertIn("print('hello')", stored_messages[0].content)
        
        self.assertIsInstance(stored_messages[1], AIMessage)
        self.assertEqual(stored_messages[1].content, "Hello User")

    def test_add_messages_sequential(self):
        history = JuChatMessageHistory(
            self.session_manager, 
            self.notebook_id, 
            self.cell_id, 
            self.current_code
        )
        
        # Add Human Message
        history.add_messages([HumanMessage(content="Request 1")])
        
        # At this point, nothing should be saved yet because we wait for AI message
        # (Based on current implementation logic)
        self.assertEqual(len(history.messages), 0)
        
        # Add AI Message
        history.add_messages([AIMessage(content="Response 1")])
        
        # Now it should be saved
        stored_messages = history.messages
        self.assertEqual(len(stored_messages), 2)
        
        self.assertIn("Intent: Request 1", stored_messages[0].content)
        self.assertEqual(stored_messages[1].content, "Response 1")

if __name__ == "__main__":
    unittest.main()
