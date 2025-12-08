import logging
import os
import sys
from logging.handlers import RotatingFileHandler

# 日志格式
DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

def setup_logging(log_dir="logs"):
    """
    配置日志系统，分离运行时日志和LLM交互日志。
    """
    # 确保日志目录存在
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir)
        except Exception as e:
            print(f"Failed to create log directory {log_dir}: {e}", file=sys.stderr)
            return

    # --- 1. 配置通用运行时日志 (Jupyter Runtime) ---
    # 获取 'aiserver' 命名空间下的根 logger
    runtime_logger = logging.getLogger("aiserver")
    runtime_logger.setLevel(logging.INFO)
    runtime_logger.propagate = False  # 防止传播到 Jupyter Server 的根 logger，避免重复或混杂

    # 避免重复添加 Handler (在热重载时可能发生)
    if not runtime_logger.handlers:
        # File Handler: aiserver_runtime.log
        runtime_file = os.path.join(log_dir, "aiserver_runtime.log")
        runtime_file_handler = RotatingFileHandler(
            runtime_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
        )
        runtime_file_handler.setFormatter(logging.Formatter(DEFAULT_FORMAT))
        runtime_logger.addHandler(runtime_file_handler)
        
        # Console Handler: 输出到控制台
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(DEFAULT_FORMAT))
        runtime_logger.addHandler(console_handler)

    # --- 2. 配置 LLM 交互日志 (LLM Interaction) ---
    # 使用 'aiserver.llm' 作为专用命名空间
    llm_logger = logging.getLogger("aiserver.llm")
    llm_logger.setLevel(logging.INFO)
    llm_logger.propagate = False # 独立管理，不传播到 aiserver 或 root

    if not llm_logger.handlers:
        # File Handler: llm_interactions.log
        llm_file = os.path.join(log_dir, "llm_interactions.log")
        llm_file_handler = RotatingFileHandler(
            llm_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
        )
        llm_file_handler.setFormatter(logging.Formatter(DEFAULT_FORMAT))
        llm_logger.addHandler(llm_file_handler)
        
        # Console Handler: 可选，如果希望在控制台也看到 LLM 交互，可以添加
        # 这里我们也添加，以便调试
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(DEFAULT_FORMAT))
        llm_logger.addHandler(console_handler)

    return runtime_logger, llm_logger

def get_llm_logger():
    """获取 LLM 专用 Logger"""
    return logging.getLogger("aiserver.llm")
