import json
import tornado
import logging
import importlib
import sys
from datetime import datetime
from jupyter_server.base.handlers import APIHandler
from jupyter_server.utils import url_path_join
from deepseek import DeepSeekClient
from .config import ConfigManager
from . import system_prompts as prompts
from .user_prompts import construct_user_prompt

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deepseek_debug.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RouteHandler(APIHandler):
    # Removed @tornado.web.authenticated decorator to disable authentication
    def get(self):
        self.finish(json.dumps({
            "data": "This is /aiserver/get-example endpoint!"
        }))


class GenerateHandler(APIHandler):
    def initialize(self):
        """初始化生成服务处理器

        从后端配置管理器读取默认提供商参数（API Key、模型等），
        不再使用 .env 环境变量作为回退。
        """
        try:
            cm = ConfigManager()
            cfg = cm.get_config()
            provider_name = cfg.default_provider
            provider_cfg = cm.get_provider_config(provider_name)

            if provider_name == "deepseek" and provider_cfg and provider_cfg.api_key:
                self.deepseek_client = DeepSeekClient(api_key=provider_cfg.api_key)
                logger.info("DeepSeek客户端初始化成功 (ConfigManager)")
            else:
                self.deepseek_client = None
                logger.warning("DeepSeek客户端初始化失败：未找到API密钥 (ConfigManager)")
        except Exception as e:
            self.deepseek_client = None
            logger.error(f"读取后端配置失败: {e}")
    
    # Removed @tornado.web.authenticated decorator to disable authentication
    def post(self):
        try:
            # Parse the request body
            data = self.get_json_body()
            logger.info(f"收到生成请求: {json.dumps(data, ensure_ascii=False, indent=2)}")
            
            # Extract parameters
            language = data.get("language", "python")
            source = data.get("source", "")
            context = data.get("context", {})
            intent = data.get("intent", "")
            options = data.get("options", {})
            output = data.get("output", "")  # 获取代码执行输出（错误信息等）
            variables = data.get("variables", []) # 获取变量信息
            
            logger.info(f"请求参数 - 语言: {language}, 意图: {intent}, 模式: {options.get('mode', 'create')}, 变量数: {len(variables)}")
            
            # Generate code suggestion based on the provided parameters
            suggestion = self.generate_suggestion(language, source, context, intent, options, output, variables)
            
            logger.info(f"生成的建议长度: {len(suggestion)} 字符")
            
            # Send the response
            response = json.dumps({
                "suggestion": suggestion,
                "explanation": "这是一个由AI助手生成的代码建议"
            })
            logger.info("成功返回AI建议")
            self.finish(response)
        except Exception as e:
            logger.error(f"AI服务调用异常: {str(e)}", exc_info=True)
            self.set_status(500)
            self.finish(json.dumps({
                "error": f"AI服务调用失败: {str(e)}",
                "suggestion": "# 示例代码:\nprint('Hello from AI Assistant!')",
                "explanation": "AI服务暂时不可用，请稍后再试"
            }))
    
    def generate_suggestion(self, language, source, context, intent, options, output=None, variables=None):
        """
        Generate code suggestion based on the provided parameters using DeepSeek API.
        """
        # 如果没有配置API密钥，则返回示例代码
        if not self.deepseek_client:
            logger.warning("DeepSeek客户端未初始化，返回示例代码")
            # 示例实现 - 实际情况下这里会调用AI服务
            if intent:
                return f"# 根据您的描述 '{intent}' 生成的代码:\nprint('Hello from AI Assistant!')"
            else:
                return "# 示例代码:\nprint('Hello from AI Assistant!')"
        
        # 构造系统提示词和用户提示词
        system_prompt = self.construct_system_prompt(options)
        user_prompt = construct_user_prompt(language, source, context, intent, options, output, variables)
        
        logger.info(f"系统提示词长度: {len(system_prompt)} 字符")
        logger.info(f"完整系统提示词:\n{system_prompt}")

        logger.info(f"用户提示词长度: {len(user_prompt)} 字符")
        logger.info(f"完整用户提示词:\n{user_prompt}")
        
        # 调用DeepSeek API
        try:
            logger.info("开始调用DeepSeek API...")
            start_time = datetime.now()
            
            request_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            
            logger.info(f"请求参数 - 模型: deepseek-coder, 消息数量: {len(request_messages)}, 流式: False")
            
            response = self.deepseek_client.chat_completion(
                model="deepseek-coder",
                messages=request_messages,
                stream=False
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info(f"DeepSeek API调用完成，耗时: {duration:.2f}秒")
            logger.info(f"API响应状态: {response}")
            
            # 提取生成的代码
            suggestion = response.choices[0].message.content
            logger.info(f"原始响应内容长度: {len(suggestion)} 字符")
            logger.debug(f"原始响应内容:\n{suggestion}")
            
            # 根据模式决定是否清理 Markdown 代码块标记
            mode = options.get("mode", "create")
            
            if mode != "explain":
                # create/fix 模式：移除可能的 Markdown 代码块标记
                if suggestion.startswith("```"):
                    logger.info("检测到 Markdown 代码块标记，正在清理...")
                    lines = suggestion.split("\n")
                    if len(lines) > 2:
                        suggestion = "\n".join(lines[1:-1])
                        logger.info(f"清理后内容长度: {len(suggestion)} 字符")
                        logger.debug(f"清理后内容:\n{suggestion}")
            
            return suggestion
        except Exception as e:
            logger.error(f"DeepSeek API调用异常: {str(e)}", exc_info=True)
            # 如果API调用失败，返回错误信息和示例代码
            return f"# AI服务调用失败: {str(e)}\n# 示例代码:\nprint('Hello from AI Assistant!')"
    
    def jls_extract_def(self, context, prompt_parts):
        if context and (context.get("prev") or context.get("next")):
            prompt_parts.append("\n相关上下文:")
            context_info = "添加相关上下文: "
            if context.get("prev"):
                prev_info = f"之前代码 {len(context['prev'])} 个单元"
                prompt_parts.append("之前的代码:")
                for i, code in enumerate(context["prev"]):
                    prompt_parts.append(f"  [{i+1}] {code}")
                context_info += prev_info
            if context.get("next"):
                next_info = f"之后代码 {len(context['next'])} 个单元"
                prompt_parts.append("之后的代码:")
                for i, code in enumerate(context["next"]):
                    prompt_parts.append(f"  [{i+1}] {code}")
                if context.get("prev"):
                    context_info += ", "
                context_info += next_info
            logger.info(context_info)
        else:
            logger.info("未提供邻近单元格上下文")
        
        return context_info

    def construct_system_prompt(self, options):
        """
        构造系统提示词，定义AI的角色和输出规范。
        
        Args:
            options: 选项字典，包含 mode 等参数
            
        Returns:
            str: 系统提示词内容
        """
        mode = options.get("mode", "create")
        logger.info(f"构造系统提示词，模式: {mode}")
        
        # 直接获取模式对应的完整系统提示词
        system_content = prompts.MODE_PROMPTS.get(mode, prompts.MODE_PROMPTS["create"])
        
        logger.info(f"系统提示词构造完成，长度: {len(system_content)} 字符")
        return system_content
    


class AnalyzeDataFrameHandler(APIHandler):
    def initialize(self):
        """初始化数据分析处理器

        从后端配置管理器读取默认提供商配置，不再使用 .env 回退。
        """
        try:
            cm = ConfigManager()
            cfg = cm.get_config()
            provider_name = cfg.default_provider
            provider_cfg = cm.get_provider_config(provider_name)

            if provider_name == "deepseek" and provider_cfg and provider_cfg.api_key:
                self.deepseek_client = DeepSeekClient(api_key=provider_cfg.api_key)
                logger.info("DeepSeek客户端初始化成功 (ConfigManager)")
            else:
                self.deepseek_client = None
                logger.warning("DeepSeek客户端初始化失败：未找到API密钥 (ConfigManager)")
        except Exception as e:
            self.deepseek_client = None
            logger.error(f"读取后端配置失败: {e}")
    
    # Removed @tornado.web.authenticated decorator to disable authentication
    def post(self):
        try:
            # Parse the request body
            data = self.get_json_body()
            logger.info(f"收到数据分析请求: {json.dumps(data, ensure_ascii=False, indent=2)}")
            
            # Extract parameters
            df_name = data.get("dfName", "")
            metadata = data.get("metadata", {})
            intent = data.get("intent", "")
            
            logger.info(f"请求参数 - DataFrame名称: {df_name}, 意图: {intent}")
            
            # Generate data analysis code based on the provided parameters
            suggestion = self.generate_analysis_code(df_name, metadata, intent)
            
            logger.info(f"生成的分析代码长度: {len(suggestion)} 字符")
            
            # Send the response
            response = json.dumps({
                "suggestion": suggestion,
                "explanation": "这是一个由AI助手生成的数据分析代码"
            })
            logger.info("成功返回数据分析代码")
            self.finish(response)
        except Exception as e:
            logger.error(f"数据分析服务调用异常: {str(e)}", exc_info=True)
            self.set_status(500)
            self.finish(json.dumps({
                "error": f"数据分析服务调用失败: {str(e)}",
                "suggestion": "# 示例代码:\nprint(df.head())",
                "explanation": "数据分析服务暂时不可用，请稍后再试"
            }))
    
    def generate_analysis_code(self, df_name, metadata, intent):
        """
        Generate data analysis code based on the provided parameters using DeepSeek API.
        """
        # 如果没有配置API密钥，则返回示例代码
        if not self.deepseek_client:
            logger.warning("DeepSeek客户端未初始化，返回示例代码")
            # 示例实现 - 实际情况下这里会调用AI服务
            if intent:
                return f"# 根据您的描述 '{intent}' 生成的数据分析代码:\nprint(df.head())"
            else:
                return "# 示例数据分析代码:\nprint(df.head())"
        
        # 构造提示词
        prompt = self.construct_analysis_prompt(df_name, metadata, intent)
        logger.info(f"构造的数据分析提示词长度: {len(prompt)} 字符")
        logger.debug(f"完整提示词内容:\n{prompt}")
        
        # 调用DeepSeek API
        try:
            logger.info("开始调用DeepSeek API进行数据分析...")
            start_time = datetime.now()
            
            request_messages = [
                {"role": "system", "content": prompts.ANALYSIS_SYSTEM_MESSAGE},
                {"role": "user", "content": prompt}
            ]
            
            logger.info(f"请求参数 - 模型: deepseek-coder, 消息数量: {len(request_messages)}, 流式: False")
            
            response = self.deepseek_client.chat_completion(
                model="deepseek-coder",
                messages=request_messages,
                stream=False
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info(f"DeepSeek API调用完成，耗时: {duration:.2f}秒")
            logger.info(f"API响应状态: {response}")
            
            # 提取生成的代码
            suggestion = response.choices[0].message.content
            logger.info(f"原始响应内容长度: {len(suggestion)} 字符")
            logger.debug(f"原始响应内容:\n{suggestion}")
            
            # 移除可能的markdown代码块标记
            if suggestion.startswith("```"):
                logger.info("检测到markdown代码块标记，正在清理...")
                lines = suggestion.split("\n")
                if len(lines) > 2:
                    suggestion = "\n".join(lines[1:-1])
                    logger.info(f"清理后内容长度: {len(suggestion)} 字符")
                    logger.debug(f"清理后内容:\n{suggestion}")
            
            return suggestion
        except Exception as e:
            logger.error(f"DeepSeek API调用异常: {str(e)}", exc_info=True)
            # 如果API调用失败，返回错误信息和示例代码
            return f"# AI服务调用失败: {str(e)}\n# 示例代码:\nprint(df.head())"
    
    def construct_analysis_prompt(self, df_name, metadata, intent):
        """
        Construct a prompt for the AI model based on the provided parameters.
        """
        logger.info("开始构造数据分析提示词...")
        prompt_parts = []
        
        # 系统指令
        system_instruction = prompts.ANALYSIS_SYSTEM_INSTRUCTION
        prompt_parts.append(system_instruction)
        logger.info(f"添加系统指令: {system_instruction}")
        
        # 添加意图描述
        if intent:
            intent_text = f"请根据以下描述生成数据分析代码: {intent}"
            prompt_parts.append(intent_text)
            logger.info(f"添加意图描述: {intent}")
        else:
            prompt_parts.append("请生成一些有用的数据分析代码:")
            logger.info("添加默认数据分析代码生成请求")
        
        # 添加DataFrame元数据
        if metadata:
            metadata_text = "\nDataFrame元数据:"
            prompt_parts.append(metadata_text)
            
            # 添加基本信息
            if df_name:
                prompt_parts.append(f"- 名称: {df_name}")
            if metadata.get("shape"):
                prompt_parts.append(f"- 形状: {metadata['shape']}")
            if metadata.get("columns"):
                prompt_parts.append(f"- 列名: {metadata['columns']}")
            if metadata.get("dtypes"):
                prompt_parts.append(f"- 数据类型: {metadata['dtypes']}")
            
            logger.info(f"添加DataFrame元数据: {json.dumps(metadata, ensure_ascii=False)}")
        else:
            logger.info("未提供DataFrame元数据")
        
        # 添加特定指令
        prompt_parts.extend(prompts.ANALYSIS_REQUIREMENTS)
        
        logger.info("添加数据分析指令和要求")
        
        final_prompt = "\n".join(prompt_parts)
        logger.info(f"数据分析提示词构造完成，总长度: {len(final_prompt)} 字符")
        
        return final_prompt


from . import algorithm_prompts
from . import algorithm_templates

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
            
            # aiserver.algorithm_templates depends on aiserver.algorithms
            if 'aiserver.algorithm_templates' in sys.modules:
                importlib.reload(sys.modules['aiserver.algorithm_templates'])
                logger.info("Reloaded aiserver.algorithm_templates")

            # aiserver.algorithm_prompts depends on aiserver.algorithms
            if 'aiserver.algorithm_prompts' in sys.modules:
                importlib.reload(sys.modules['aiserver.algorithm_prompts'])
                logger.info("Reloaded aiserver.algorithm_prompts")
            
            # 3. Fetch new metadata
            library = algorithm_templates.get_library_metadata()
            self.finish(json.dumps(library, ensure_ascii=False))
            
        except Exception as e:
            logger.error(f"Failed to reload function library: {e}", exc_info=True)
            self.set_status(500)
            self.finish(json.dumps({"error": str(e)}))

class GetCSVColumnsHandler(APIHandler):
    # Removed @tornado.web.authenticated decorator to disable authentication
    def post(self):
        try:
            data = self.get_json_body()
            filepath = data.get("filepath", "")
            
            if not filepath:
                 raise ValueError("Filepath is required")

            # Ensure path is relative to CWD if not absolute
            if not os.path.isabs(filepath):
                filepath = os.path.join(os.getcwd(), filepath)
            
            if not os.path.exists(filepath):
                # Try relative to 'dataset' if typical pattern
                filepath_dataset = os.path.join(os.getcwd(), "dataset", os.path.basename(filepath))
                if os.path.exists(filepath_dataset):
                    filepath = filepath_dataset
                else:
                    raise FileNotFoundError(f"File not found: {filepath}")

            import pandas as pd
            # Read only header
            df = pd.read_csv(filepath, nrows=0)
            columns = df.columns.tolist()
            
            self.finish(json.dumps({"columns": columns}))
        except Exception as e:
            logger.error(f"Failed to get CSV columns: {e}")
            self.set_status(500)
            self.finish(json.dumps({"error": str(e)}))

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
from .config import ConfigManager
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
