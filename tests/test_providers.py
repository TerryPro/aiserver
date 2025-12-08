import pytest
from unittest.mock import MagicMock, patch
import sys
import os

# Ensure aiserver is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from aiserver.aiserver.providers import ProviderManager
from aiserver.aiserver.config import AppConfig, ProviderConfig

@pytest.fixture
def mock_config_manager():
    with patch('aiserver.aiserver.providers.ConfigManager') as MockConfigManager:
        instance = MockConfigManager.return_value
        
        # Setup default config
        config = AppConfig(
            default_provider="deepseek",
            providers={
                "deepseek": ProviderConfig(
                    api_key="test-key",
                    base_url="https://test.url",
                    model="deepseek-test",
                    enabled=True
                ),
                "openai": ProviderConfig(
                    api_key="openai-key",
                    model="gpt-4",
                    enabled=True
                ),
                "disabled-provider": ProviderConfig(
                    api_key="key",
                    model="model",
                    enabled=False
                )
            }
        )
        instance.get_config.return_value = config
        yield instance

def test_get_provider_default(mock_config_manager):
    manager = ProviderManager()
    manager.refresh() # Ensure clean state
    
    with patch('aiserver.aiserver.providers.ChatOpenAI') as MockChatOpenAI:
        provider = manager.get_provider()
        
        # assert provider == MockChatOpenAI.return_value
        MockChatOpenAI.assert_called_with(
            model="deepseek-test",
            api_key="test-key",
            base_url="https://test.url",
            temperature=0.1,
            max_tokens=2000,
            streaming=True
        )

def test_get_provider_openai(mock_config_manager):
    manager = ProviderManager()
    manager.refresh()
    
    with patch('aiserver.aiserver.providers.ChatOpenAI') as MockChatOpenAI:
        provider = manager.get_provider("openai")
        
        MockChatOpenAI.assert_called_with(
            model="gpt-4",
            api_key="openai-key",
            base_url=None,
            temperature=0.1,
            max_tokens=2000,
            streaming=True
        )

def test_get_provider_disabled(mock_config_manager):
    manager = ProviderManager()
    manager.refresh()
    
    with pytest.raises(ValueError, match="is disabled"):
        manager.get_provider("disabled-provider")

def test_get_provider_missing(mock_config_manager):
    manager = ProviderManager()
    manager.refresh()
    
    with pytest.raises(ValueError, match="not found"):
        manager.get_provider("non-existent")
