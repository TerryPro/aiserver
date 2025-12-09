import json
import logging
from jupyter_server.base.handlers import APIHandler
from tornado import web
from ..core.providers import ProviderManager

logger = logging.getLogger(__name__)

class ModelsHandler(APIHandler):
    """
    Handler for managing AI models.
    """
    
    def initialize(self):
        try:
            self.provider_manager = ProviderManager()
        except Exception as e:
            logger.error(f"Failed to initialize Provider Manager: {e}")

    @web.authenticated
    def get(self):
        """
        Get available models and the current default model.
        """
        try:
            config = self.provider_manager.config_manager.get_config()
            
            # List all configured providers/models
            models = []
            for name, provider_config in config.providers.items():
                if provider_config.enabled:
                    models.append({
                        "id": name,
                        "name": name, # Display name could be improved
                        "model": provider_config.model,
                        "isDefault": name == config.default_provider
                    })
            
            response = {
                "current": config.default_provider,
                "models": models
            }
            
            self.finish(json.dumps(response))
            
        except Exception as e:
            logger.error(f"Error retrieving models: {e}")
            self.set_status(500)
            self.finish(json.dumps({"error": str(e)}))

    @web.authenticated
    def post(self):
        """
        Set the current default model.
        """
        try:
            data = self.get_json_body()
            model_id = data.get("model_id")
            
            if not model_id:
                self.set_status(400)
                self.finish(json.dumps({"error": "Missing model_id parameter"}))
                return

            config = self.provider_manager.config_manager.get_config()
            
            if model_id not in config.providers:
                self.set_status(404)
                self.finish(json.dumps({"error": f"Model {model_id} not found"}))
                return
                
            if not config.providers[model_id].enabled:
                self.set_status(400)
                self.finish(json.dumps({"error": f"Model {model_id} is disabled"}))
                return
            
            # Update default provider
            # Note: We are modifying the config object directly. 
            # In a production environment with concurrency, we might need locking.
            config.default_provider = model_id
            
            # Persist changes
            self.provider_manager.config_manager.save_config()
            
            # Refresh provider manager to ensure next request picks up changes if needed
            # (Though ProviderManager uses config_manager.get_config() each time usually)
            self.provider_manager.refresh()
            
            self.finish(json.dumps({"status": "success", "current": model_id}))
            
        except Exception as e:
            logger.error(f"Error setting model: {e}")
            self.set_status(500)
            self.finish(json.dumps({"error": str(e)}))
