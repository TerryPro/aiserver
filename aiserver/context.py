"""
Context Manager for AI Server
Handles variable formatting, prompt construction, and token management.
"""
from typing import Dict, List, Any, Optional
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage

class ContextManager:
    """
    Manages context for AI interactions, including variable formatting and prompt construction.
    """
    
    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens

    def format_variables(self, variables: Optional[Dict[str, Any]]) -> str:
        """
        Format a dictionary of variables into a context string.
        
        Args:
            variables: Dictionary mapping variable names to their values or metadata.
            
        Returns:
            Formatted context string.
        """
        if not variables:
            return "No variables available."
            
        context_lines = ["Active Variables:"]
        for name, value in variables.items():
            # Handle dictionary with metadata (type, value, etc.) if provided
            if isinstance(value, dict) and 'type' in value and 'value' in value:
                var_type = value.get('type', 'unknown')
                var_value = value.get('value', '')
                line = f"- {name} ({var_type}): {self._truncate(str(var_value))}"
            else:
                # Simple value
                line = f"- {name}: {self._truncate(str(value))}"
            context_lines.append(line)
            
        return "\n".join(context_lines)

    def _truncate(self, text: str, max_length: int = 200) -> str:
        """Truncate text to max_length."""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "... (truncated)"

    def construct_messages(
        self, 
        system_prompt: str, 
        user_query: str, 
        variable_context: str, 
        history: List[Dict[str, str]]
    ) -> List[BaseMessage]:
        """
        Construct the list of messages for the LLM.
        
        Args:
            system_prompt: The base system instruction.
            user_query: The current user message.
            variable_context: Formatted variable context string.
            history: List of previous messages ({'role': 'user'/'assistant', 'content': '...'}).
            
        Returns:
            List of LangChain BaseMessage objects.
        """
        messages: List[BaseMessage] = []
        
        # 1. System Message with Context
        full_system_content = f"{system_prompt}\n\n{variable_context}"
        messages.append(SystemMessage(content=full_system_content))
        
        # 2. History
        for msg in history:
            role = msg.get('role')
            content = msg.get('content')
            if role == 'user':
                messages.append(HumanMessage(content=content))
            elif role == 'assistant':
                messages.append(AIMessage(content=content))
                
        # 3. Current User Message
        messages.append(HumanMessage(content=user_query))
        
        return messages

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count (approximate).
        For production, use tiktoken or provider-specific tokenizers.
        """
        # Rough estimation: 1 token ~= 4 chars (English) or 0.7 chars (Chinese)
        # Using a conservative average for mixed content
        return len(text) // 3

    def trim_history(self, history: List[Dict[str, str]], max_history_tokens: int) -> List[Dict[str, str]]:
        """
        Trim history to fit within token limits.
        Removes oldest messages first.
        """
        current_tokens = sum(self.count_tokens(msg['content']) for msg in history)
        
        while current_tokens > max_history_tokens and history:
            removed = history.pop(0)
            current_tokens -= self.count_tokens(removed['content'])
            
        return history
