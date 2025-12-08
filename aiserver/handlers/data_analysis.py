import json
import logging
import os
from datetime import datetime
import pandas as pd
from jupyter_server.base.handlers import APIHandler
from langchain_core.messages import SystemMessage, HumanMessage
from ..core.providers import ProviderManager
from ..prompts import system as prompts

logger = logging.getLogger(__name__)

class AnalyzeDataFrameHandler(APIHandler):
    def initialize(self):
        """初始化数据分析处理器

        从后端配置管理器读取默认提供商配置，不再使用 .env 回退。
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
        Generate data analysis code based on the provided parameters using LangChain.
        """
        # 获取 LLM 提供者
        try:
            llm = self.provider_manager.get_provider()
        except Exception as e:
            logger.warning(f"LLM provider not available: {e}")
            if intent:
                return f"# 根据您的描述 '{intent}' 生成的数据分析代码:\nprint(df.head())"
            else:
                return "# 示例数据分析代码:\nprint(df.head())"
        
        # 构造提示词
        prompt = self.construct_analysis_prompt(df_name, metadata, intent)
        logger.info(f"构造的数据分析提示词长度: {len(prompt)} 字符")
        logger.debug(f"完整提示词内容:\n{prompt}")
        
        # 调用 AI Provider
        try:
            logger.info("开始调用 AI Provider 进行数据分析...")
            start_time = datetime.now()
            
            messages = [
                SystemMessage(content=prompts.ANALYSIS_SYSTEM_MESSAGE),
                HumanMessage(content=prompt)
            ]
            
            logger.info(f"请求消息数量: {len(messages)}")
            
            response = llm.invoke(messages)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info(f"AI 调用完成，耗时: {duration:.2f}秒")
            logger.info(f"API响应: {response}")
            
            # 提取生成的代码
            suggestion = response.content
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
            logger.error(f"AI服务调用异常: {str(e)}", exc_info=True)
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

            # Read only header
            df = pd.read_csv(filepath, nrows=0)
            columns = df.columns.tolist()
            
            self.finish(json.dumps({"columns": columns}))
        except Exception as e:
            logger.error(f"Failed to get CSV columns: {e}")
            self.set_status(500)
            self.finish(json.dumps({"error": str(e)}))
