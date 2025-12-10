# Summary Generation Prompts

SUMMARY_PROMPT_DETAILED = """
请对本次代码生成操作进行总结。
用户意图: {intent}
当前模式: {mode}
生成代码片段: {code_snippet}...

请输出合法的 JSON 格式，包含两个字段：
1. "summary": 简短的一句话总结（30字以内），不包含主语，例如"已生成读取CSV的代码"。
2. "detailed_summary": 详细的技术总结，包含"实现内容"、"修复内容"或者"修改错误"（根据意图编写）。请根据代码变更推断实现细节。

示例输出:
{{
  "summary": "已添加数据预处理逻辑",
  "detailed_summary": "实现内容\\n1.使用 Pandas 读取 CSV...\\n 2.清洗了空值..."
}}

注意：请确保输出是标准的 JSON 格式，不要包含 Markdown 代码块标记（如 ```json）。
"""

SUMMARY_PROMPT_SIMPLE = """
请用一句简短的中文（30字以内）总结以下操作。
当前模式: {mode}
用户意图: {intent}
生成结果片段: {code_snippet}...

例如: "已解释数据清洗逻辑"。
不要包含"用户"、"AI"等主语，直接陈述动作。
请输出 JSON 格式: {{"summary": "你的总结", "detailed_summary": ""}}
"""
