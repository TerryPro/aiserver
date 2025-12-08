import json
import logging
import importlib
import sys
from jupyter_server.base.handlers import APIHandler
from ..prompts import algorithm as algorithm_prompts
from ..lib import library as algorithm_templates

logger = logging.getLogger(__name__)

class GetFunctionLibraryHandler(APIHandler):
    # Removed @tornado.web.authenticated decorator to disable authentication
    def get(self):
        try:
            library = algorithm_templates.get_library_metadata()
            self.finish(json.dumps(library, ensure_ascii=False))
        except Exception as e:
            logger.error(f"Failed to fetch function library: {e}")
            self.set_status(500)
            self.finish(json.dumps({"error": str(e)}))

class GetAlgorithmPromptsHandler(APIHandler):
    # Removed @tornado.web.authenticated decorator to disable authentication
    def get(self):
        self.finish(json.dumps(algorithm_prompts.ALGORITHM_PROMPTS, ensure_ascii=False))

class ReloadFunctionLibraryHandler(APIHandler):
    # Removed @tornado.web.authenticated decorator to disable authentication
    def post(self):
        try:
            logger.info("Reloading function library...")
            
            # 1. Reload base library modules
            # We need to reload submodules first, then the package
            submodules = [
                'algorithm.data_loading',
                'algorithm.data_operation',
                'algorithm.data_preprocessing',
                'algorithm.eda',
                'algorithm.anomaly_detection',
                'algorithm.trend',
                'algorithm.plotting',
            ]
            
            for module_name in submodules:
                if module_name in sys.modules:
                    try:
                        importlib.reload(sys.modules[module_name])
                        logger.info(f"Reloaded {module_name}")
                    except Exception as e:
                        logger.warning(f"Failed to reload {module_name}: {e}")

            if 'algorithm' in sys.modules:
                importlib.reload(sys.modules['algorithm'])
                logger.info("Reloaded algorithm package")

            # 2. Reload aiserver adapters
            # aiserver.algorithms depends on algorithm
            if 'aiserver.algorithms' in sys.modules:
                importlib.reload(sys.modules['aiserver.algorithms'])
                logger.info("Reloaded aiserver.algorithms")
            
            # aiserver.lib.library depends on aiserver.algorithms
            if 'aiserver.lib.library' in sys.modules:
                importlib.reload(sys.modules['aiserver.lib.library'])
                logger.info("Reloaded aiserver.lib.library")

            # aiserver.prompts.algorithm depends on aiserver.algorithms
            if 'aiserver.prompts.algorithm' in sys.modules:
                importlib.reload(sys.modules['aiserver.prompts.algorithm'])
                logger.info("Reloaded aiserver.prompts.algorithm")
            
            # 3. Fetch new metadata
            library = algorithm_templates.get_library_metadata()
            self.finish(json.dumps(library, ensure_ascii=False))
            
        except Exception as e:
            logger.error(f"Failed to reload function library: {e}", exc_info=True)
            self.set_status(500)
            self.finish(json.dumps({"error": str(e)}))
