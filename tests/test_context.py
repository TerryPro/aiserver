import pytest
from aiserver.aiserver.context import ContextManager
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

def test_format_variables():
    cm = ContextManager()
    variables = {
        "df": {"type": "DataFrame", "value": "columns: [a, b], rows: 5"},
        "x": 10,
        "long_str": "a" * 300
    }
    context = cm.format_variables(variables)
    assert "Active Variables:" in context
    assert "df (DataFrame): columns: [a, b], rows: 5" in context
    assert "x: 10" in context
    assert "... (truncated)" in context

def test_construct_messages():
    cm = ContextManager()
    system_prompt = "You are a helper."
    user_query = "Help me."
    variable_context = "Var: x=1"
    history = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"}
    ]
    
    messages = cm.construct_messages(system_prompt, user_query, variable_context, history)
    
    assert len(messages) == 4
    assert isinstance(messages[0], SystemMessage)
    assert "You are a helper." in messages[0].content
    assert "Var: x=1" in messages[0].content
    assert isinstance(messages[1], HumanMessage)
    assert messages[1].content == "Hi"
    assert isinstance(messages[2], AIMessage)
    assert messages[2].content == "Hello"
    assert isinstance(messages[3], HumanMessage)
    assert messages[3].content == "Help me."

def test_trim_history():
    cm = ContextManager()
    # "123456" (6 chars) -> 2 tokens
    # "123" (3 chars) -> 1 token
    history = [
        {"role": "user", "content": "123456"}, 
        {"role": "assistant", "content": "123"} 
    ]
    
    # Total tokens approx 3. Limit to 1.
    trimmed = cm.trim_history(history[:], 1)
    
    assert len(trimmed) == 1
    assert trimmed[0]['content'] == "123"
