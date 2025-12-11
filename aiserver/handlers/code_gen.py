import json
import logging
from datetime import datetime
from jupyter_server.base.handlers import APIHandler
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from ..core.providers import ProviderManager
from ..core.session import SessionManager
from ..core.history import JuChatMessageHistory
from ..core.context import ContextOptimizer
from ..prompts import system as prompts
from ..prompts.user import construct_user_prompt
from ..prompts.templates import get_chat_prompt
from ..prompts import summary as summary_prompts
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

        try:
            import inspect
            logger.info(f"DEBUG: SessionManager loaded from {inspect.getfile(SessionManager)}")
            logger.info(f"DEBUG: SessionManager methods: {dir(SessionManager)}")
            self.ai_session_manager = SessionManager()
            self.context_optimizer = ContextOptimizer()
            logger.info("Session Manager & Context Optimizer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Session Manager: {e}")
    
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
            notebook_id = data.get("notebookId")
            cell_id = data.get("cellId")
            
            # Get session history
            history = []
            if notebook_id and cell_id:
                try:
                    history = self.ai_session_manager.get_history(notebook_id, cell_id)
                    llm_logger.info(f"Retrieved {len(history)} history messages for cell {cell_id}")
                except Exception as e:
                    logger.warning(f"Failed to get session history: {e}")

            # Generate code suggestion based on the provided parameters
            suggestion = self.generate_suggestion(
                language, source, context, intent, options, output, variables, history,
                notebook_id=notebook_id, cell_id=cell_id, data=data
            )
                      
            # Task 5: Generate Summary & Update History
            summary = None
            detailed_summary = None
            if notebook_id and cell_id:
                try:
                    # 获取 LLM 实例用于生成总结
                    llm = self.provider_manager.get_provider()
                    summary_result = self._generate_summary(llm, intent, suggestion, options.get("mode"))
                    
                    if isinstance(summary_result, dict):
                        summary = summary_result.get("summary")
                        detailed_summary = summary_result.get("detailed_summary")
                    else:
                        summary = summary_result
                    
                    # 更新历史记录中的 summary
                    self.ai_session_manager.update_last_interaction_summary(notebook_id, cell_id, summary, detailed_summary)
                    llm_logger.info(f"已生成并保存总结: {summary}")
                except Exception as e:
                    llm_logger.warning(f"生成或保存总结失败: {e}")

            # Send the response
            response_data = {
                "suggestion": suggestion,
                "explanation": "这是一个由AI助手生成的代码建议"
            }
            if summary:
                response_data["summary"] = summary
            if detailed_summary:
                response_data["detailed_summary"] = detailed_summary
                
            response = json.dumps(response_data)
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
    
    def generate_suggestion(self, language, source, context, intent, options, output=None, variables=None, history=None, notebook_id=None, cell_id=None, data=None):
        """
        Generate code suggestion based on the provided parameters using LangChain.
        """
        if data is None:
            data = {}
        # 获取 LLM 提供者
        try:
            llm = self.provider_manager.get_provider()
        except Exception as e:
            logger.warning(f"LLM provider not available: {e}")
            if intent:
                return f"# 根据您的描述 '{intent}' 生成的代码:\nprint('Hello from AI Assistant!')"
            else:
                return "# 示例代码:\nprint('Hello from AI Assistant!')"
        
        # 1. 构造 Prompt 并计算 Token 预算
        system_prompt = self.construct_system_prompt(options)
        
        # Token 估算与分配
        # 假设总限制 8000，预留 1000 给输出
        MAX_TOTAL = self.context_optimizer.max_total_tokens
        RESERVED_OUTPUT = 1000
        
        system_tokens = self.context_optimizer.estimate_tokens(system_prompt)
        source_tokens = self.context_optimizer.estimate_tokens(source)
        
        available_tokens = MAX_TOTAL - RESERVED_OUTPUT - system_tokens - source_tokens
        
        # 动态分配 Context 和 History 的 Token 限制
        # 默认上限: Context 3000, History 2000
        context_limit = 3000
        history_limit = 2000
        
        if available_tokens < 100:
            # 空间极度紧张，禁用上下文和历史
            context_limit = 0
            history_limit = 0
            llm_logger.warning(f"Token budget extremely low ({available_tokens}). Disabling context and history.")
        elif available_tokens < (context_limit + history_limit):
            # 空间不足，按比例缩减 (Context 60%, History 40%)
            context_limit = int(available_tokens * 0.6)
            history_limit = int(available_tokens * 0.4)
            llm_logger.info(f"Token budget tight ({available_tokens}). Adjusting limits: Context={context_limit}, History={history_limit}")
            
        # 更新 Optimizer 配置
        self.context_optimizer.max_history_tokens = history_limit
        
        # 提取跨Cell上下文 (传入限制)
        include_context = data.get("include_context", False)
        cross_cell_context = ""
        if include_context:
            cross_cell_context = self._extract_cross_cell_context(context, max_tokens=context_limit)
        
        # 注意：这里我们暂时仍然使用 construct_user_prompt 来生成 User Input 字符串，
        # 但在后续任务中，我们可以将其拆解为 Prompt Template 的一部分
        user_input_content = construct_user_prompt(language, source, context, intent, options, output, variables, history=None) # History passed as None to avoid duplication in user prompt
        
        if cross_cell_context:
            user_input_content = cross_cell_context + "\n" + user_input_content
            
        prompt_template = get_chat_prompt(system_prompt)
        
        llm_logger.info(f"系统提示词长度: {len(system_prompt)} 字符")
        llm_logger.info(f"完整系统提示词:\n{system_prompt}")

        llm_logger.info(f"用户输入内容长度: {len(user_input_content)} 字符")
        llm_logger.info(f"完整用户输入内容:\n{user_input_content}")
        
        # 2. 构建 Chain (LCEL)
        # Prompt -> LLM -> OutputParser
        chain = prompt_template | llm | StrOutputParser()
        
        # 调用 AI Provider
        try:
            llm_logger.info("开始调用 AI Provider (LangChain)...")
            start_time = datetime.now()
            
            # 3. 执行 Chain
            if notebook_id and cell_id:
                # 使用 RunnableWithMessageHistory 管理历史记录
                def get_session_history(session_id: str) -> JuChatMessageHistory:
                    # session_id format: "notebook_id::cell_id"
                    return JuChatMessageHistory(
                        session_manager=self.ai_session_manager,
                        notebook_id=notebook_id,
                        cell_id=cell_id,
                        current_code=source,
                        optimizer=self.context_optimizer
                    )
                
                chain_with_history = RunnableWithMessageHistory(
                    chain,
                    get_session_history,
                    input_messages_key="intent",
                    history_messages_key="history",
                )
                
                # "input" is used by the Prompt Template (user_input_content with code context)
                # "intent" is used by RunnableWithMessageHistory to save the user message to history
                response_content = chain_with_history.invoke(
                    {
                        "input": user_input_content,
                        "intent": intent or "Continued Conversation"
                    },
                    config={"configurable": {"session_id": f"{notebook_id}::{cell_id}"}}
                )
            else:
                # 无会话信息，不使用历史记录
                response_content = chain.invoke({
                    "history": [], 
                    "input": user_input_content
                })
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            llm_logger.info(f"AI 调用完成，耗时: {duration:.2f}秒")
            
            # 提取生成的代码 (StrOutputParser 已经提取了 content)
            suggestion = response_content
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

    def _extract_cross_cell_context(self, context, max_tokens=3000):
        """
        提取跨Cell的上下文信息（AST解析）
        
        Args:
            context: 前端传入的 context 字典，包含 prev/next 代码列表
            max_tokens: 最大 Token 限制
            
        Returns:
            str: 格式化的上下文信息字符串
        """
        if not context or not context.get("prev"):
            return ""
            
        prev_codes = context.get("prev", [])
        return self.context_optimizer.optimize_code_context(prev_codes, max_tokens=max_tokens)

    def _generate_summary(self, llm, intent, code_suggestion, mode):
        """
        生成本次操作的总结 (Task 5)
        
        Args:
            llm: LLM Provider 实例
            intent: 用户意图
            code_suggestion: 生成的代码建议
            mode: 当前模式
            
        Returns:
            dict: 包含 'summary' (简短) 和 'detailed_summary' (详细) 的字典
        """
        if not intent:
            return {"summary": "已根据上下文生成代码"}
            
        try:
            # 构造 Prompt
            if mode in ["create", "fix", "refactor"]:
                summary_prompt_content = summary_prompts.SUMMARY_PROMPT_DETAILED.format(
                    intent=intent,
                    mode=mode,
                    code_snippet=code_suggestion[:500]
                )
            else:
                # explain 模式或其他模式保持简单
                summary_prompt_content = summary_prompts.SUMMARY_PROMPT_SIMPLE.format(
                    intent=intent,
                    mode=mode,
                    code_snippet=code_suggestion[:200]
                )
            
            # 直接调用 LLM
            response = llm.invoke([HumanMessage(content=summary_prompt_content)])
            
            # 处理响应
            content = ""
            if isinstance(response, str):
                content = response
            elif hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
            
            content = content.strip()
            # 清理可能的 Markdown 标记
            if content.startswith("```"):
                lines = content.split("\n")
                if len(lines) > 2:
                    content = "\n".join(lines[1:-1])
            
            import json
            try:
                result = json.loads(content)
                if not isinstance(result, dict):
                    # 如果解析出来不是字典，回退
                    return {"summary": str(result).strip()[:30]}
                return result
            except json.JSONDecodeError:
                # 如果 JSON 解析失败，尝试直接提取文本作为简短总结
                return {"summary": content.strip()[:30]}
            
        except Exception as e:
            llm_logger.warning(f"总结生成失败: {e}")
            return {"summary": "操作已完成"}

    def construct_system_prompt(self, options):
        """
        构造系统提示词，定义AI的角色和输出规范。
        
        Args:
            options: 选项字典，包含 mode 等参数
            
        Returns:
            str: 系统提示词内容
        """
        mode = options.get("mode", "create")
        
        # 直接获取模式对应的完整系统提示词
        system_content = prompts.MODE_PROMPTS.get(mode, prompts.MODE_PROMPTS["create"])
        
        return system_content
