import json
import tornado
import os
import logging
from datetime import datetime
from dotenv import load_dotenv
from jupyter_server.base.handlers import APIHandler
from jupyter_server.utils import url_path_join
from deepseek import DeepSeekClient

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
        # Load environment variables from .env file
        load_dotenv()
        
        # Initialize DeepSeek client
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        self.deepseek_client = DeepSeekClient(api_key=api_key) if api_key else None
        
        # 记录客户端初始化信息
        if self.deepseek_client:
            logger.info("DeepSeek客户端初始化成功")
        else:
            logger.warning("DeepSeek客户端初始化失败：未找到API密钥")
    
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
            
            logger.info(f"请求参数 - 语言: {language}, 意图: {intent}, 模式: {options.get('mode', 'replace')}, 变量数: {len(variables)}")
            
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
        user_prompt = self.construct_user_prompt(language, source, context, intent, options, output, variables)
        
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
            mode = options.get("mode", "replace")
            
            if mode != "explain":
                # replace/fix 模式：移除可能的 Markdown 代码块标记
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
        mode = options.get("mode", "replace")
        logger.info(f"构造系统提示词，模式: {mode}")
        
        # 基础角色定义
        base_role = "你是一个专业的Jupyter Notebook开发助手。"
        
        # 通用编码规范
        common_rules = [
            "编码规范：",
            "- 遵循Python最佳实践（PEP 8）",
            "- 遵循Jupyter Notebook最佳实践",
            "- 使用matplotlib绘制图表",
            "- 使用清晰、描述性的变量名",
            "- 对关键步骤添加中文注释",
            "- 确保代码简洁高效"
        ]
        
        # 根据模式构造不同的提示词
        if mode == "explain":
            prompt_parts = [
                base_role,
                "",
                "当前任务：根据当前代码生成Markdown格式的代码解释文档",
                "",
                "输出要求：",
                "- 使用标准Markdown格式（标题、列表、代码引用等）",
                "- 文档内容全面准确，易于理解",
                "- 解释代码的目的、原理和关键逻辑",
                "",
                *common_rules
            ]
        elif mode == "fix":
            prompt_parts = [
                base_role,
                "",
                "当前任务：根据用户意图分析并修复代码中的问题或者优化代码",
                "",
                "输出要求：",
                "- 只输出Python代码和注释，不要包含任何Markdown标记",
                "- 绝对不要使用```python或```等代码块标记",
                "- 直接输出可执行的Python代码，不要有任何格式包装",
                "- 代码必须完整且独立，可直接在Jupyter Notebook中运行",
                "",
                "修复指导：",
                "- 仔细分析执行结果中的错误信息",
                "- 保留原代码中正确的部分，只修复有问题的地方",
                "- 如果是语法错误，修正语法问题",
                "- 如果是逻辑错误，调整代码逻辑",
                "- 如果是运行时错误，添加必要的错误处理或数据验证",
                "- 在修复的地方添加注释说明修改原因",
                "",
                *common_rules
            ]
        else:  # replace 模式
            prompt_parts = [
                base_role,
                "",
                "当前任务：根据用户意图编写Python代码",
                "",
                "输出要求：",
                "- 只输出Python代码和注释，不要包含任何Markdown标记",
                "- 绝对不要使用```python或```等代码块标记",
                "- 直接输出可执行的Python代码，不要有任何格式包装",
                "- 代码必须完整且独立，可直接在Jupyter Notebook中运行",
                "",
                "生成指导：",
                "- 充分理解用户意图，生成符合需求的代码",
                "- 如果提供了环境变量（DataFrame等），优先使用这些变量",
                "- 代码应该具有良好的可读性和可维护性",
                "- 对复杂逻辑添加详细的中文注释",
                "",
                *common_rules
            ]
        
        system_content = "\n".join(prompt_parts)
        logger.info(f"系统提示词构造完成，长度: {len(system_content)} 字符")
        return system_content
    
    def construct_user_prompt(self, language, source, context, intent, options, output=None, variables=None):
        """
        构造用户提示词，包含具体任务和上下文数据。
        
        Args:
            language: 编程语言
            source: 当前代码
            context: 上下文信息
            intent: 用户意图
            options: 选项字典
            output: 代码执行输出
            variables: 环境变量列表
            
        Returns:
            str: 用户提示词内容
        """
        logger.info("开始构造用户提示词...")
        prompt_parts = []
        
        # 1. 任务描述
        mode = options.get("mode", "replace")
        logger.info(f"设置生成模式: {mode}")
        
        if mode == "replace":
            task_desc = "根据用户意图和环境变量，生成Python代码"
        elif mode == "fix":
            task_desc = "根据用户意图、当前代码和执行结果，修复代码问题"
        elif mode == "explain":
            task_desc = "对当前代码和执行结果进行详细解释"
        else:
            task_desc = "生成代码"
        
        prompt_parts.append(f"任务：{task_desc}")
        logger.info(f"添加任务描述: {task_desc}")
        
        # 2. 用户意图
        if intent:
            prompt_parts.append(f"\n用户意图：\n{intent}")
            logger.info(f"添加用户意图，长度: {len(intent)} 字符")
        
        # 3. 环境变量（DataFrame）
        if variables:
            prompt_parts.append("\n环境变量（DataFrame）：")
            for var in variables:
                prompt_parts.append(f"- 变量名: {var.get('name')}")
                prompt_parts.append(f"  - 形状: {var.get('shape')}")
                prompt_parts.append(f"  - 列名: {var.get('columns')}")
                prompt_parts.append(f"  - 类型: {var.get('dtypes')}")
            logger.info(f"添加 {len(variables)} 个环境变量")
        
        # 4. 当前代码
        if source:
            prompt_parts.append(f"\n当前代码：\n{source}")
            logger.info(f"添加当前代码，长度: {len(source)} 字符")
        
        # 5. 执行结果/错误信息
        if output:
            prompt_parts.append(f"\n执行结果：\n{output}")
            logger.info(f"添加执行结果，长度: {len(output)} 字符")
        
        final_prompt = "\n".join(prompt_parts)
        logger.info(f"提示词构造完成，总长度: {len(final_prompt)} 字符")
        
        return final_prompt


class AnalyzeDataFrameHandler(APIHandler):
    def initialize(self):
        # Load environment variables from .env file
        load_dotenv()
        
        # Initialize DeepSeek client
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        self.deepseek_client = DeepSeekClient(api_key=api_key) if api_key else None
        
        # 记录客户端初始化信息
        if self.deepseek_client:
            logger.info("DeepSeek客户端初始化成功")
        else:
            logger.warning("DeepSeek客户端初始化失败：未找到API密钥")
    
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
                {"role": "system", "content": "你是一个专业的数据分析助手，专门生成可在Jupyter Notebook中直接执行的Python数据分析代码。确保生成的代码符合Python规范，包含适当的中文注释，并且不包含任何markdown代码块标记（如```python）。"},
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
        system_instruction = "你是一个专业的数据分析助手，擅长生成可在Jupyter Notebook中直接执行的Python数据分析代码。"
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
        prompt_parts.append("\n要求：")
        prompt_parts.append("1. 生成的Python代码必须能在Jupyter Notebook的代码单元格中直接执行")
        prompt_parts.append("2. 包含适当的中文注释，解释关键步骤")
        prompt_parts.append("3. 不要包含任何markdown代码块标记（如```python）")
        prompt_parts.append("4. 代码应该是完整且独立的")
        prompt_parts.append("5. 使用pandas库进行数据分析")
        prompt_parts.append("6. 如果需要可视化，使用matplotlib或seaborn库")
        
        logger.info("添加数据分析指令和要求")
        
        final_prompt = "\n".join(prompt_parts)
        logger.info(f"数据分析提示词构造完成，总长度: {len(final_prompt)} 字符")
        
        return final_prompt


def setup_handlers(web_app):
    host_pattern = ".*$"

    base_url = web_app.settings["base_url"]
    handlers = [
        (url_path_join(base_url, "aiserver", "get-example"), RouteHandler),
        (url_path_join(base_url, "aiserver", "generate"), GenerateHandler),
        (url_path_join(base_url, "aiserver", "analyze-dataframe"), AnalyzeDataFrameHandler)
    ]
    web_app.add_handlers(host_pattern, handlers)
