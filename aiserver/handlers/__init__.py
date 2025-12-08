import logging
from jupyter_server.utils import url_path_join
from .code_gen import GenerateHandler
from .data_analysis import AnalyzeDataFrameHandler, GetCSVColumnsHandler
from .library import GetFunctionLibraryHandler, GetAlgorithmPromptsHandler, ReloadFunctionLibraryHandler
from .system import GetServerRootHandler, RouteHandler, ConfigHandler

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deepseek_debug.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def setup_handlers(web_app):
    host_pattern = ".*$"
    
    base_url = web_app.settings["base_url"]
    
    # 定义路由
    handlers = [
        (url_path_join(base_url, "aiserver", "generate"), GenerateHandler),
        (url_path_join(base_url, "aiserver", "analyze-dataframe"), AnalyzeDataFrameHandler),
        (url_path_join(base_url, "aiserver", "algorithm-prompts"), GetAlgorithmPromptsHandler),
        (url_path_join(base_url, "aiserver", "function-library"), GetFunctionLibraryHandler),
        (url_path_join(base_url, "aiserver", "reload-library"), ReloadFunctionLibraryHandler),
        (url_path_join(base_url, "aiserver", "get-csv-columns"), GetCSVColumnsHandler),
        (url_path_join(base_url, "aiserver", "get-server-root"), GetServerRootHandler),
        (url_path_join(base_url, "aiserver", "get-example"), RouteHandler),
        (url_path_join(base_url, "aiserver", "config"), ConfigHandler)
    ]
    
    web_app.add_handlers(host_pattern, handlers)
