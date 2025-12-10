import unittest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from aiserver.handlers.code_gen import GenerateHandler

class TestCodeGenChain(unittest.TestCase):
    @patch('aiserver.handlers.code_gen.ProviderManager')
    @patch('aiserver.handlers.code_gen.get_chat_prompt')
    @patch('aiserver.handlers.code_gen.construct_user_prompt')
    def test_generate_suggestion_chain(self, mock_construct_user_prompt, mock_get_chat_prompt, mock_provider_manager):
        # Setup mocks
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="print('Hello World')")
        
        # Mock provider manager instance
        mock_pm_instance = MagicMock()
        mock_pm_instance.get_provider.return_value = mock_llm
        mock_provider_manager.return_value = mock_pm_instance
        
        # Mock prompts
        mock_construct_user_prompt.return_value = "User Input"
        
        # Mock ChatPromptTemplate and Chain
        mock_prompt_template = MagicMock(spec=ChatPromptTemplate)
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "print('Hello World')"
        
        # This is the tricky part: we need to mock the pipe operator |
        # prompt | llm | parser
        # We can mock the return value of the pipe operations
        mock_prompt_template.__or__.return_value = MagicMock()
        mock_prompt_template.__or__.return_value.__or__.return_value = mock_chain
        
        mock_get_chat_prompt.return_value = mock_prompt_template

        # Create a partial mock of GenerateHandler to avoid Tornado init
        # We can't easily instantiate APIHandler without a valid application/request
        # So we'll dynamically create a class or just patch the method's 'self'
        
        # Alternative: Subclass and override __init__
        class TestHandler(GenerateHandler):
            def __init__(self):
                self.provider_manager = mock_pm_instance
                self.context_optimizer = MagicMock()
                self.context_optimizer.max_total_tokens = 8000
                self.context_optimizer.estimate_tokens.return_value = 10 # Mock token count
                self.context_optimizer.optimize_code_context.return_value = "" # Mock optimization
            
            # Mock construct_system_prompt as it's a method on self
            def construct_system_prompt(self, options):
                return "System Prompt"

        handler = TestHandler()
        
        # Call generate_suggestion
        result = handler.generate_suggestion(
            language="python",
            source="print('hi')",
            context={},
            intent="test",
            options={"mode": "create"},
            output="",
            variables=[],
            history=None
        )
        
        # Verification
        self.assertEqual(result, "print('Hello World')")
        
        # Verify get_chat_prompt called with system prompt
        mock_get_chat_prompt.assert_called_with("System Prompt")
        
        # Verify construct_user_prompt called
        mock_construct_user_prompt.assert_called()
        
        # Verify chain.invoke called with correct inputs
        # Note: We can't easily verify the chain structure strictly without using real LangChain objects,
        # but we can verify that the final chain object we mocked was invoked.
        mock_chain.invoke.assert_called_with({
            "history": [],
            "input": "User Input"
        })

    @patch('aiserver.handlers.code_gen.RunnableWithMessageHistory')
    @patch('aiserver.handlers.code_gen.JuChatMessageHistory')
    @patch('aiserver.handlers.code_gen.ProviderManager')
    @patch('aiserver.handlers.code_gen.get_chat_prompt')
    @patch('aiserver.handlers.code_gen.construct_user_prompt')
    def test_generate_suggestion_with_history(self, mock_construct_user_prompt, mock_get_chat_prompt, mock_provider_manager, mock_ju_history, mock_runnable_history):
        # Setup mocks similar to previous test
        mock_llm = MagicMock()
        mock_pm_instance = MagicMock()
        mock_pm_instance.get_provider.return_value = mock_llm
        mock_provider_manager.return_value = mock_pm_instance
        
        mock_construct_user_prompt.return_value = "User Input with Context"
        
        mock_prompt_template = MagicMock(spec=ChatPromptTemplate)
        mock_chain = MagicMock()
        mock_prompt_template.__or__.return_value = MagicMock()
        mock_prompt_template.__or__.return_value.__or__.return_value = mock_chain
        
        mock_get_chat_prompt.return_value = mock_prompt_template

        # Mock RunnableWithMessageHistory
        mock_chain_with_history = MagicMock()
        mock_chain_with_history.invoke.return_value = "AI Response"
        mock_runnable_history.return_value = mock_chain_with_history

        # Mock Handler
        class TestHandler(GenerateHandler):
            def __init__(self):
                self.provider_manager = mock_pm_instance
                self.ai_session_manager = MagicMock() # Mock session manager
                self.context_optimizer = MagicMock()
                self.context_optimizer.max_total_tokens = 8000
                self.context_optimizer.estimate_tokens.return_value = 10
                self.context_optimizer.optimize_code_context.return_value = ""
            
            def construct_system_prompt(self, options):
                return "System Prompt"

        handler = TestHandler()
        
        # Call generate_suggestion with notebook_id and cell_id
        result = handler.generate_suggestion(
            language="python",
            source="print('hi')",
            context={},
            intent="Fix this",
            options={"mode": "fix"},
            output="",
            variables=[],
            history=None,
            notebook_id="nb1",
            cell_id="cell1"
        )
        
        # Verification
        self.assertEqual(result, "AI Response")
        
        # Verify RunnableWithMessageHistory created
        mock_runnable_history.assert_called()
        args, kwargs = mock_runnable_history.call_args
        self.assertEqual(kwargs['input_messages_key'], "intent")
        self.assertEqual(kwargs['history_messages_key'], "history")
        
        # Verify get_session_history factory logic
        # We can retrieve the factory function passed to RunnableWithMessageHistory
        factory = args[1]
        # Call factory to verify it creates JuChatMessageHistory correctly
        history_instance = factory("nb1::cell1")
        mock_ju_history.assert_called_with(
            session_manager=handler.ai_session_manager,
            notebook_id="nb1",
            cell_id="cell1",
            current_code="print('hi')",
            optimizer=handler.context_optimizer
        )
        
        # Verify chain_with_history.invoke called with correct config and input
        mock_chain_with_history.invoke.assert_called_with(
            {
                "input": "User Input with Context",
                "intent": "Fix this"
            },
            config={"configurable": {"session_id": "nb1::cell1"}}
        )

    def test_generate_summary(self):
        """Test _generate_summary method with new prompts"""
        # Setup
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content='{"summary": "test summary", "detailed_summary": "detailed"}')
        
        # Create a partial mock of GenerateHandler
        # We use __new__ to avoid __init__ which requires Tornado app context
        handler = GenerateHandler.__new__(GenerateHandler)
        
        # Test create mode (Detailed Prompt)
        result = handler._generate_summary(mock_llm, "test intent", "print('hello')", "create")
        self.assertEqual(result["summary"], "test summary")
        self.assertEqual(result["detailed_summary"], "detailed")
        
        # Verify detailed prompt was used
        mock_llm.invoke.assert_called()
        call_arg = mock_llm.invoke.call_args[0][0][0] # First arg, first element (list), first item (HumanMessage)
        self.assertIn("用户意图: test intent", call_arg.content)
        self.assertIn("当前模式: create", call_arg.content)
        
        # Test explain mode (Simple Prompt)
        mock_llm.invoke.reset_mock()
        mock_llm.invoke.return_value = AIMessage(content='{"summary": "simple summary", "detailed_summary": ""}')
        
        result = handler._generate_summary(mock_llm, "explain code", "print('hello')", "explain")
        self.assertEqual(result["summary"], "simple summary")
        
        # Verify simple prompt was used
        call_arg = mock_llm.invoke.call_args[0][0][0]
        self.assertIn("当前模式: explain", call_arg.content)
        self.assertIn("用户意图: explain code", call_arg.content)

if __name__ == '__main__':
    unittest.main()
