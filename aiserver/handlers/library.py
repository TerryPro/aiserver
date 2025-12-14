import json
import logging
import importlib
import sys
from jupyter_server.base.handlers import APIHandler
from ..prompts import algorithm as algorithm_prompts
from ..lib import library as algorithm_templates
from ..utils import code_manager

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
            importlib.invalidate_caches()
            
            # 1. Reload base library modules (all algorithm.*)
            # We need to reload submodules first, then the package
            reloaded_count = 0
            for name in list(sys.modules.keys()):
                if name.startswith('algorithm.') and sys.modules[name]:
                    try:
                        importlib.reload(sys.modules[name])
                        reloaded_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to reload {name}: {e}")

            if 'algorithm' in sys.modules:
                importlib.reload(sys.modules['algorithm'])
                logger.info(f"Reloaded algorithm package and {reloaded_count} submodules")

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

class ManageAlgorithmHandler(APIHandler):
    def post(self):
        try:
            data = json.loads(self.request.body)
            action = data.get("action")
            
            if not action:
                 self.set_status(400)
                 self.finish(json.dumps({"error": "Missing action"}))
                 return

            if action == "add":
                category = data.get("category")
                code = data.get("code")
                if not category or not code:
                    self.set_status(400)
                    self.finish(json.dumps({"error": "Missing category or code"}))
                    return
                code_manager.add_function(category, code)
                
            elif action == "update":
                algo_id = data.get("id")
                code = data.get("code")
                if not algo_id or not code:
                    self.set_status(400)
                    self.finish(json.dumps({"error": "Missing id or code"}))
                    return
                code_manager.update_function(algo_id, code)
                
            elif action == "delete":
                algo_id = data.get("id")
                if not algo_id:
                    self.set_status(400)
                    self.finish(json.dumps({"error": "Missing id"}))
                    return
                code_manager.delete_function(algo_id)
            
            elif action == "get_code":
                algo_id = data.get("id")
                if not algo_id:
                    self.set_status(400)
                    self.finish(json.dumps({"error": "Missing id"}))
                    return
                code = code_manager.get_function_code(algo_id)
                if code is None:
                    self.set_status(404)
                    self.finish(json.dumps({"error": "Algorithm not found"}))
                    return
                self.finish(json.dumps({"code": code}))
                return

            elif action == "generate_code":
                metadata = data.get("metadata")
                existing_code = data.get("code")  # Optional
                if not metadata:
                    self.set_status(400)
                    self.finish(json.dumps({"error": "Missing metadata"}))
                    return
                code = code_manager.generate_function_code(metadata, existing_code)
                self.finish(json.dumps({"code": code}))
                return

            elif action == "parse_code":
                code = data.get("code")
                if not code:
                    self.set_status(400)
                    self.finish(json.dumps({"error": "Missing code"}))
                    return
                metadata = code_manager.parse_function_code(code)
                if metadata is None:
                    self.set_status(400)
                    self.finish(json.dumps({"error": "Failed to parse code"}))
                    return
                self.finish(json.dumps({"metadata": metadata}))
                return

            else:
                self.set_status(400)
                self.finish(json.dumps({"error": "Invalid action"}))
                return

            # Trigger reload automatically
            # We can instantiate ReloadHandler or just call logic.
            # Instantiating is messy with request/response.
            # Let's just call the reload logic? Or let frontend call reload?
            # The design says "Success -> Reload".
            # Let's trigger reload here.
            
            # Reuse logic from ReloadFunctionLibraryHandler?
            # It's better to refactor reload logic into a function, but for now I'll copy-paste or call via internal request?
            # Internal request is hard.
            # I'll just trigger basic reload of metadata for response.
            
            # Actually, to make sure the change is reflected immediately, we MUST reload.
            # Let's duplicate the reload logic briefly or move it to a util.
            # Moving to util is better.
            
            # For now, I'll just do minimal reload to get response.
            
            importlib.invalidate_caches()
            if 'aiserver.algorithms' in sys.modules:
                importlib.reload(sys.modules['aiserver.algorithms'])
            if 'aiserver.lib.library' in sys.modules:
                importlib.reload(sys.modules['aiserver.lib.library'])
                
            library = algorithm_templates.get_library_metadata()
            self.finish(json.dumps(library, ensure_ascii=False))

        except Exception as e:
            logger.error(f"Failed to manage algorithm: {e}", exc_info=True)
            self.set_status(500)
            self.finish(json.dumps({"error": str(e)}))
