import logging

from ..core.log import get_llm_logger

llm_logger = get_llm_logger()

def construct_user_prompt(language, source, context, intent, options, output=None, variables=None, history=None):
    prompt_parts = []
    
    # Add history if provided
    if history:
        prompt_parts.append("# 历史对话 (Conversation History)")
        prompt_parts.append("这是针对当前代码单元的过往对话记录，请参考这些上下文来理解用户的最新意图。")
        prompt_parts.append("<HISTORY>")
        if isinstance(history, list):
            for msg in history:
                if hasattr(msg, 'type') and hasattr(msg, 'content'):
                    role = "AI" if msg.type == 'ai' else "User"
                    prompt_parts.append(f"[{role}]: {msg.content}")
                else:
                    prompt_parts.append(str(msg))
        else:
            prompt_parts.append(str(history))
        prompt_parts.append("</HISTORY>\n")

    prompt_parts.append("# 用户意图 (User Intent)")
    if intent:
        prompt_parts.append(intent)
    else:
        prompt_parts.append("（未提供具体意图）")
    prompt_parts.append("\n# 环境变量 (Current Data Context)")
    prompt_parts.append("以下是当前 Jupyter 环境中存在的关键变量及其结构（DataFrame形状、列名和类型等）。请在编写或修改代码时参考并优先使用这些变量。")
    if variables:
        prompt_parts.append("\n<VARIABLES>")
        for var in variables:
            prompt_parts.append(f"- 变量名: {var.get('name')}")
            prompt_parts.append(f"  - 形状: {var.get('shape')}")
            prompt_parts.append(f"  - 列名: {var.get('columns')}")
            prompt_parts.append(f"  - 类型: {var.get('dtypes')}")
        prompt_parts.append("\n</VARIABLES>")
    else:
        prompt_parts.append("（当前没有可用的环境变量）")
    prompt_parts.append("\n# 当前代码 (Code to be Fixed/Optimized)")
    prompt_parts.append("请严格解析并处理下方 START 和 END 标记之间的完整 Python 代码。")
    prompt_parts.append("\n<CODE>")
    prompt_parts.append(source if source else "")
    prompt_parts.append("\n</CODE>")
    prompt_parts.append("\n# 执行结果 (Execution Result)")
    prompt_parts.append("请参考此处的输出，尤其是 Traceback 和错误信息。如果代码运行成功，请留空。")
    prompt_parts.append("\n<OUTPUT>")
    prompt_parts.append(output if output else "")
    prompt_parts.append("\n</OUTPUT>")
    final_prompt = "\n".join(prompt_parts)
    llm_logger.info(f"提示词构造完成，总长度: {len(final_prompt)} 字符")
    return final_prompt

