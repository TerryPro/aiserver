import os
import sys
import json
import pytest
from unittest.mock import patch, mock_open

# Ensure module can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../aiserver')))

from config import ConfigManager, AppConfig, ProviderConfig

# Mock jupyter_config_dir to avoid messing with real user config
@pytest.fixture
def mock_config_dir(tmp_path):
    # Because we are running tests with 'from config import ...', 
    # we should patch 'config.jupyter_config_dir' instead of 'aiserver.config.jupyter_config_dir'
    with patch("config.jupyter_config_dir", return_value=str(tmp_path)):
        yield tmp_path

@pytest.fixture
def config_manager(mock_config_dir):
    # Reset singleton
    ConfigManager._instance = None
    return ConfigManager()

def test_default_config(config_manager):
    """Test that default config is loaded when file doesn't exist"""
    config = config_manager.get_config()
    assert config.default_provider == "deepseek"
    assert "deepseek" in config.providers
    assert config.providers["deepseek"].model == "deepseek-coder"

def test_ignore_env_vars(mock_config_dir):
    """Default config should NOT read environment variables"""
    with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key-123"}):
        ConfigManager._instance = None
        cm = ConfigManager()
        config = cm.get_config()
        assert config.providers["deepseek"].api_key is None

def test_save_and_load_config(config_manager):
    """Test persistence"""
    # Modify config
    config_manager.update_config({
        "default_provider": "openai",
        "providers": {
            "openai": {"api_key": "sk-test", "model": "gpt-3.5-turbo", "enabled": True}
        }
    })
    
    # Check in-memory update
    config = config_manager.get_config()
    assert config.default_provider == "openai"
    assert config.providers["openai"].api_key == "sk-test"
    
    # Check file on disk
    config_file = config_manager.config_file
    assert config_file.exists()
    
    with open(config_file, 'r') as f:
        saved_data = json.load(f)
        assert saved_data["default_provider"] == "openai"
        assert saved_data["providers"]["openai"]["api_key"] == "sk-test"

    # Reload from disk
    ConfigManager._instance = None
    new_cm = ConfigManager()
    loaded_config = new_cm.get_config()
    assert loaded_config.default_provider == "openai"
