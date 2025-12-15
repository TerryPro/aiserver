import json
import logging
import importlib
import sys
from jupyter_server.base.handlers import APIHandler
from ..prompts import algorithm as algorithm_prompts
from ..lib import library as algorithm_templates
from ..utils import code_manager
from ..utils.reload_helper import reload_algorithm_modules

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
            
            # 使用统一的重载工具
            reload_result = reload_algorithm_modules()
            
            # 获取新的元数据
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

            # 触发自动重载
            reload_algorithm_modules()
            
            # 获取更新后的元数据
            library = algorithm_templates.get_library_metadata()
            self.finish(json.dumps(library, ensure_ascii=False))

        except Exception as e:
            logger.error(f"Failed to manage algorithm: {e}", exc_info=True)
            self.set_status(500)
            self.finish(json.dumps({"error": str(e)}))
