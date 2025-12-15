import logging
from jupyter_server.utils import url_path_join
from .code_gen import GenerateHandler
from .data_analysis import AnalyzeDataFrameHandler, GetCSVColumnsHandler
from .library import GetFunctionLibraryHandler, GetAlgorithmPromptsHandler, ReloadFunctionLibraryHandler, ManageAlgorithmHandler
from .system import GetServerRootHandler, RouteHandler, ConfigHandler
from .session import SessionHistoryHandler
from .models import ModelsHandler
from .algorithm_save import SaveAlgorithmHandler
from ..core.log import setup_logging

# 配置日志系统
setup_logging()

def setup_handlers(web_app):
    host_pattern = ".*$"
    
    base_url = web_app.settings["base_url"]
    
    # 定义路由
    handlers = [
        (url_path_join(base_url, "aiserver", "generate"), GenerateHandler),
        (url_path_join(base_url, "aiserver", "sessions"), SessionHistoryHandler),
        (url_path_join(base_url, "aiserver", "models"), ModelsHandler),
        (url_path_join(base_url, "aiserver", "analyze-dataframe"), AnalyzeDataFrameHandler),
        (url_path_join(base_url, "aiserver", "algorithm-prompts"), GetAlgorithmPromptsHandler),
        (url_path_join(base_url, "aiserver", "function-library"), GetFunctionLibraryHandler),
        (url_path_join(base_url, "aiserver", "reload-library"), ReloadFunctionLibraryHandler),
        (url_path_join(base_url, "aiserver", "algorithm-manage"), ManageAlgorithmHandler),
        (url_path_join(base_url, "aiserver", "algorithm-save"), SaveAlgorithmHandler),
        (url_path_join(base_url, "aiserver", "get-csv-columns"), GetCSVColumnsHandler),
        (url_path_join(base_url, "aiserver", "get-server-root"), GetServerRootHandler),
        (url_path_join(base_url, "aiserver", "get-example"), RouteHandler),
        (url_path_join(base_url, "aiserver", "config"), ConfigHandler)
    ]
    
    web_app.add_handlers(host_pattern, handlers)
