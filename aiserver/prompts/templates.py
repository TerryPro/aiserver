from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# System Prompt Templates for ChatPromptTemplate
# We reuse the content from prompts/system.py but adapt it for ChatPromptTemplate

def get_chat_prompt(system_prompt_content: str) -> ChatPromptTemplate:
    """
    Constructs a ChatPromptTemplate with:
    1. System Prompt (from argument)
    2. History Placeholder (optional, for future memory integration)
    3. Human Message (User Input)
    """
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt_content),
        MessagesPlaceholder(variable_name="history", optional=True),
        HumanMessagePromptTemplate.from_template("{input}")
    ])
