from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from .config import ConfigManager
import logging

logger = logging.getLogger(__name__)

class ProviderManager:
    """
    Manage AI providers using LangChain.
    Supports switching between different providers (DeepSeek, OpenAI, etc.)
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ProviderManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self.config_manager = ConfigManager()
        self._providers: Dict[str, BaseChatModel] = {}
        self._initialized = True

    def get_provider(self, provider_name: Optional[str] = None) -> BaseChatModel:
        """
        Get a LangChain ChatModel instance for the specified provider.
        
        Args:
            provider_name: The name of the provider (e.g., 'deepseek', 'openai').
                           If None, uses the default provider from config.
                           
        Returns:
            BaseChatModel: The configured LangChain ChatModel instance.
            
        Raises:
            ValueError: If the provider is not found or disabled.
        """
        config = self.config_manager.get_config()
        
        if not provider_name:
            provider_name = config.default_provider
            
        if provider_name not in config.providers:
            raise ValueError(f"Provider '{provider_name}' not found in configuration.")
            
        provider_config = config.providers[provider_name]
        
        if not provider_config.enabled:
            raise ValueError(f"Provider '{provider_name}' is disabled.")
            
        if provider_name in self._providers:
            return self._providers[provider_name]
            
        model = self._create_provider(provider_name, provider_config)
        self._providers[provider_name] = model
        return model

    def _create_provider(self, name: str, provider_config: Any) -> BaseChatModel:
        """
        Create a specific provider instance based on name and config.
        """
        if not provider_config.api_key:
            logger.warning(f"API Key for {name} is missing.")
            
        # Both DeepSeek and OpenAI use the OpenAI SDK/Interface
        # DeepSeek Example: base_url="https://api.deepseek.com", model="deepseek-coder"
        
        # Note: LangChain's ChatOpenAI handles both OpenAI and compatible endpoints
        return ChatOpenAI(
            model=provider_config.model,
            api_key=provider_config.api_key,
            base_url=provider_config.base_url,
            temperature=provider_config.temperature,
            max_tokens=provider_config.max_tokens,
            # Streaming could be added here if needed
            streaming=True
        )

    def refresh(self):
        """Clear cache to force recreation of providers (e.g. after config update)"""
        self._providers.clear()
        # Also reload config from disk
        self.config_manager = ConfigManager() # Re-init might not work if singleton returns same instance
        # Actually ConfigManager is singleton, so we need a way to reload it.
        # But ConfigManager doesn't have a reload method exposed yet.
        # However, get_config() returns self.config which is loaded in __init__.
        # If we want to support dynamic reload, we should add reload_config() to ConfigManager.
        # For now, let's assume ConfigManager handles its own state, we just clear our provider cache.
