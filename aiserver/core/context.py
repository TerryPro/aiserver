import logging
from typing import List, Dict, Optional, Union
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from ..utils.code_analysis import extract_code_metadata

logger = logging.getLogger(__name__)

class ContextOptimizer:
    """
    上下文优化器
    负责管理和优化发送给 LLM 的上下文，包括：
    1. Token 估算
    2. 代码上下文压缩 (保留最近的完整代码，旧代码仅保留元数据)
    3. 历史记录裁剪
    """
    
    def __init__(self, max_total_tokens: int = 8000, max_history_tokens: int = 2000):
        self.max_total_tokens = max_total_tokens
        self.max_history_tokens = max_history_tokens
        
    def estimate_tokens(self, text: str) -> int:
        """
        估算文本的 Token 数量 (粗略估算：1 Token ≈ 4 字符)
        """
        if not text:
            return 0
        return len(text) // 4

    def optimize_code_context(self, prev_codes: List[str], max_tokens: int = 3000) -> str:
        """
        优化前序代码上下文
        策略：
        1. 最近的代码单元（例如最后1-2个）保留完整代码
        2. 较旧的代码单元仅保留元数据（变量、函数、类定义）
        3. 确保不超过 max_tokens
        """
        if not prev_codes:
            return ""
            
        optimized_parts = []
        current_tokens = 0
        
        # 倒序处理，优先保留最近的代码
        for i, code in enumerate(reversed(prev_codes)):
            if not code or not code.strip():
                continue
                
            cell_index = len(prev_codes) - 1 - i
            part_content = ""
            
            # 最近的 2 个单元格，尝试保留完整代码
            if i < 2:
                if len(code) < 2000: # 如果单格太大，也强制压缩
                    part_content = f"\n# Cell {cell_index}\n{code}\n"
                else:
                    part_content = self._compress_code(code, cell_index)
            else:
                # 较旧的单元格，压缩为元数据
                part_content = self._compress_code(code, cell_index)
            
            part_tokens = self.estimate_tokens(part_content)
            
            if current_tokens + part_tokens > max_tokens:
                logger.info(f"Code context limit reached at cell {cell_index}")
                break
                
            optimized_parts.insert(0, part_content)
            current_tokens += part_tokens
            
        if not optimized_parts:
            return ""
            
        return "\n<PREVIOUS_CODE>\n" + "".join(optimized_parts) + "</PREVIOUS_CODE>\n"

    def _compress_code(self, code: str, cell_index: int) -> str:
        """将代码压缩为元数据摘要"""
        metadata = extract_code_metadata(code)
        summary_lines = [f"# Cell {cell_index} (Summary)"]
        
        if metadata["variables"]:
            vars_str = ", ".join([v["name"] for v in metadata["variables"]])
            summary_lines.append(f"# Variables: {vars_str}")
            
        if metadata["functions"]:
            for f in metadata["functions"]:
                args = ", ".join(f["args"])
                doc = f.get('doc') or ""
                summary_lines.append(f"def {f['name']}({args}): ... # {doc.strip()[:50]}")
                
        if metadata["classes"]:
            for c in metadata["classes"]:
                doc = c.get('doc') or ""
                summary_lines.append(f"class {c['name']}: ... # {doc.strip()[:50]}")
        
        # 如果没有任何元数据，且代码不长，保留前几行
        if not (metadata["variables"] or metadata["functions"] or metadata["classes"]):
            lines = code.split('\n')
            preview = "\n".join(lines[:3])
            if len(lines) > 3:
                preview += "\n..."
            summary_lines.append(preview)
            
        return "\n".join(summary_lines) + "\n" + "-"*20 + "\n"

    def optimize_history(self, history: List[BaseMessage]) -> List[BaseMessage]:
        """
        裁剪历史记录以适应 Token 限制
        保留系统消息和最近的消息
        """
        if not history:
            return []
            
        optimized_history = []
        current_tokens = 0
        
        # 始终保留系统消息（如果有）
        # 注意：在 LangChain history 中通常不包含 SystemMessage，它是在 PromptTemplate 中添加的
        # 但如果 history 中包含，我们需要保留
        
        # 倒序遍历消息
        for msg in reversed(history):
            content = msg.content
            if isinstance(content, str):
                tokens = self.estimate_tokens(content)
            else:
                tokens = 0 # 忽略复杂内容
            
            if current_tokens + tokens > self.max_history_tokens:
                break
                
            optimized_history.insert(0, msg)
            current_tokens += tokens
            
        return optimized_history
