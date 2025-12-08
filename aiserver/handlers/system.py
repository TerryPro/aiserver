import json
import logging
import os
from jupyter_server.base.handlers import APIHandler
from ..core.config import ConfigManager

logger = logging.getLogger(__name__)

class GetServerRootHandler(APIHandler):
    # Removed @tornado.web.authenticated decorator to disable authentication
    def get(self):
        try:
            # Jupyter Server is typically started from the project root
            server_root = os.getcwd()
            self.finish(json.dumps({"serverRoot": server_root}))
        except Exception as e:
            logger.error(f"Failed to get server root: {e}")
            self.set_status(500)
            self.finish(json.dumps({"error": str(e)}))

class RouteHandler(APIHandler):
    # Removed @tornado.web.authenticated decorator to disable authentication
    def get(self):
        self.finish(json.dumps({
            "data": "This is /aiserver/get-example endpoint!"
        }))

class ConfigHandler(APIHandler):
    def get(self):
        """返回当前后端配置，用于前端同步显示"""
        cm = ConfigManager()
        self.finish(cm.get_config().model_dump_json())

    def put(self):
        """更新后端配置并持久化到 aiserver_config.json"""
        try:
            data = self.get_json_body() or {}
            cm = ConfigManager()
            cm.update_config(data)
            self.finish(cm.get_config().model_dump_json())
        except Exception as e:
            logger.error(f"Failed to update config: {e}")
            self.set_status(400)
            self.finish(json.dumps({"error": str(e)}))
