import json
import logging
from datetime import datetime
from jupyter_server.base.handlers import APIHandler
from langchain_core.messages import SystemMessage, HumanMessage
from ..core.providers import ProviderManager
from ..prompts import system as prompts
from ..prompts.user import construct_user_prompt
from ..core.log import get_llm_logger

logger = logging.getLogger(__name__)
llm_logger = get_llm_logger()

class GenerateHandler(APIHandler):
    def initialize(self):
        """初始化生成服务处理器

        从后端配置管理器读取默认提供商参数（API Key、模型等），
        不再使用 .env 环境变量作为回退。
        """
        try:
            self.provider_manager = ProviderManager()
            # Try to get default provider to verify it's configured
            self.provider_manager.get_provider()
            logger.info("AI Provider initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AI provider: {e}")
    
    # Removed @tornado.web.authenticated decorator to disable authentication
    def post(self):
        try:
            # Parse the request body
            data = self.get_json_body()
            llm_logger.info(f"收到生成请求: {json.dumps(data, ensure_ascii=False, indent=2)}")
            
            # Extract parameters
            language = data.get("language", "python")
            source = data.get("source", "")
            context = data.get("context", {})
            intent = data.get("intent", "")
            options = data.get("options", {})
            output = data.get("output", "")  # 获取代码执行输出（错误信息等）
            variables = data.get("variables", []) # 获取变量信息
            
            llm_logger.info(f"请求参数 - 语言: {language}, 意图: {intent}, 模式: {options.get('mode', 'create')}, 变量数: {len(variables)}")
            
            # Generate code suggestion based on the provided parameters
            suggestion = self.generate_suggestion(language, source, context, intent, options, output, variables)
            
            llm_logger.info(f"生成的建议长度: {len(suggestion)} 字符")
            
            # Send the response
            response = json.dumps({
                "suggestion": suggestion,
                "explanation": "这是一个由AI助手生成的代码建议"
            })
            llm_logger.info("成功返回AI建议")
            self.finish(response)
        except Exception as e:
            llm_logger.error(f"AI服务调用异常: {str(e)}", exc_info=True)
            self.set_status(500)
            self.finish(json.dumps({
                "error": f"AI服务调用失败: {str(e)}",
                "suggestion": "# 示例代码:\nprint('Hello from AI Assistant!')",
                "explanation": "AI服务暂时不可用，请稍后再试"
            }))
    
    def generate_suggestion(self, language, source, context, intent, options, output=None, variables=None):
        """
        Generate code suggestion based on the provided parameters using LangChain.
        """
        # 获取 LLM 提供者
        try:
            llm = self.provider_manager.get_provider()
        except Exception as e:
            logger.warning(f"LLM provider not available: {e}")
            if intent:
                return f"# 根据您的描述 '{intent}' 生成的代码:\nprint('Hello from AI Assistant!')"
            else:
                return "# 示例代码:\nprint('Hello from AI Assistant!')"
        
        # 构造系统提示词和用户提示词
        system_prompt = self.construct_system_prompt(options)
        user_prompt = construct_user_prompt(language, source, context, intent, options, output, variables)
        
        llm_logger.info(f"系统提示词长度: {len(system_prompt)} 字符")
        llm_logger.info(f"完整系统提示词:\n{system_prompt}")

        llm_logger.info(f"用户提示词长度: {len(user_prompt)} 字符")
        llm_logger.info(f"完整用户提示词:\n{user_prompt}")
        
        # 调用 AI Provider
        try:
            llm_logger.info("开始调用 AI Provider...")
            start_time = datetime.now()
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            llm_logger.info(f"请求消息数量: {len(messages)}")
            
            response = llm.invoke(messages)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            llm_logger.info(f"AI 调用完成，耗时: {duration:.2f}秒")
            llm_logger.info(f"API响应: {response}")
            
            # 提取生成的代码
            suggestion = response.content
            llm_logger.info(f"原始响应内容长度: {len(suggestion)} 字符")
            llm_logger.debug(f"原始响应内容:\n{suggestion}")
            
            # 根据模式决定是否清理 Markdown 代码块标记
            mode = options.get("mode", "create")
            
            if mode != "explain":
                # create/fix 模式：移除可能的 Markdown 代码块标记
                if suggestion.startswith("```"):
                    llm_logger.info("检测到 Markdown 代码块标记，正在清理...")
                    lines = suggestion.split("\n")
                    if len(lines) > 2:
                        suggestion = "\n".join(lines[1:-1])
                        llm_logger.info(f"清理后内容长度: {len(suggestion)} 字符")
                        llm_logger.debug(f"清理后内容:\n{suggestion}")
            
            return suggestion
        except Exception as e:
            llm_logger.error(f"AI服务调用异常: {str(e)}", exc_info=True)
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
            llm_logger.info(context_info)
        else:
            llm_logger.info("未提供邻近单元格上下文")
        
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
        llm_logger.info(f"构造系统提示词，模式: {mode}")
        
        # 直接获取模式对应的完整系统提示词
        system_content = prompts.MODE_PROMPTS.get(mode, prompts.MODE_PROMPTS["create"])
        
        llm_logger.info(f"系统提示词构造完成，长度: {len(system_content)} 字符")
        return system_content
