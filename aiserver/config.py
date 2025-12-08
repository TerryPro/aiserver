import json
import logging
from typing import Dict, Optional
from pathlib import Path
from pydantic import BaseModel, Field
from jupyter_core.paths import jupyter_config_dir

logger = logging.getLogger(__name__)

class ProviderConfig(BaseModel):
    """Configuration for a single AI provider"""
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: str
    temperature: float = 0.1
    max_tokens: int = 2000
    enabled: bool = True

class AppConfig(BaseModel):
    """Global Application Configuration"""
    default_provider: str = "deepseek"
    language: str = "en"
    
    # Provider specific settings
    providers: Dict[str, ProviderConfig] = Field(default_factory=dict)

class ConfigManager:
    """
    Singleton manager for AI Server configuration.
    
    Strategy:
    1. Try to load `aiserver_config.json` from Jupyter config directory.
    2. If not found, create default config without reading environment variables.
    3. Allow runtime updates and persistence to JSON.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.config_dir = Path(jupyter_config_dir())
        self.config_file = self.config_dir / "aiserver_config.json"
        self.config: AppConfig = self._load_config()
        self._initialized = True
        
    def _get_default_config(self) -> AppConfig:
        """Generate default config without environment variable fallback"""
        return AppConfig(
            default_provider="deepseek",
            providers={
                "deepseek": ProviderConfig(
                    api_key=None,
                    base_url="https://api.deepseek.com",
                    model="deepseek-coder",
                    enabled=True
                ),
                "openai": ProviderConfig(
                    api_key=None,
                    model="gpt-4",
                    enabled=False
                )
            }
        )

    def _load_config(self) -> AppConfig:
        """Load from JSON or return defaults"""
        if not self.config_file.exists():
            logger.info(f"Config file not found at {self.config_file}, using defaults")
            return self._get_default_config()
            
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Note: We might want to merge with defaults to ensure new fields are present
                # For simplicity, we trust the file but fall back to defaults if parsing fails
                return AppConfig(**data)
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_file}: {e}, using defaults")
            return self._get_default_config()

    def save_config(self):
        """Persist current config to disk"""
        try:
            # Ensure dir exists
            self.config_dir.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                f.write(self.config.model_dump_json(indent=2))
            logger.info(f"Config saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    def get_config(self) -> AppConfig:
        return self.config

    def get_provider_config(self, provider_name: str) -> Optional[ProviderConfig]:
        return self.config.providers.get(provider_name)

    def update_config(self, updates: Dict):
        """
        Update config with a dictionary. 
        Supports nested updates for providers.
        """
        # Handle top-level fields
        if "default_provider" in updates:
            self.config.default_provider = updates["default_provider"]
            
        # Handle providers
        if "providers" in updates:
            for name, data in updates["providers"].items():
                if name not in self.config.providers:
                    # New provider
                    # Use defaults for missing fields if creating new
                    self.config.providers[name] = ProviderConfig(**data)
                else:
                    # Update existing
                    current = self.config.providers[name]
                    # Create a copy of current data
                    current_data = current.model_dump()
                    # Update with new data
                    current_data.update(data)
                    # Re-validate
                    self.config.providers[name] = ProviderConfig(**current_data)
        
        self.save_config()
