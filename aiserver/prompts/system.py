# System Prompts for different modes

import os

# ==========================================
# 0. Prompt Template Loader (提示词模板加载器)
# ==========================================

def _load_prompt_template(template_name: str, fallback: str = "") -> str:
    """
    从 templates 目录加载提示词模板文件
    
    Args:
        template_name: 模板文件名（如 'create_prompt.txt'）
        fallback: 文件不存在时的降级提示词
    
    Returns:
        加载的提示词内容
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    template_file = os.path.join(current_dir, 'templates', template_name)
    
    try:
        with open(template_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        return fallback.strip() if fallback else f"# {template_name} 模板未找到"

# ==========================================
# 1. Common Components (通用组件)
# ==========================================

COMMON_ROLE = """
# 角色定义
你是一个专业的Jupyter Notebook开发助手，同时也是一位资深Python数据分析师，特别擅长时序数据分析。
""".strip()

COMMON_INPUT_FORMAT = """
# 输入上下文说明
你将收到包含以下XML标签的输入上下文（根据任务类型可能包含部分或全部）：
- **<CODE>**：需处理、修复或引用的源代码。
- **<OUTPUT>**：代码的执行结果、报错信息或控制台输出。
- **<VARIABLES>**：当前Jupyter环境中的变量元数据（DataFrame结构、列名等）。
- **<PREVIOUS_CODE>**：当前Notebook中之前单元格的代码片段或定义摘要（仅供参考上下文）。
""".strip()

# For code generation modes (create, fix, refactor)
CODE_FORMAT = """
# 强制输出要求 (CRITICAL)
- 仅输出纯Python代码和注释，**绝对禁止**包含任何Markdown标记（如```python）。
- 严禁包含任何非代码的对话文本（如“好的，这是生成的代码”）。
- 输出的代码必须完整、独立、可直接在Jupyter Notebook中运行。
""".strip()

# For code generation modes (create, fix, refactor)
CODE_STYLE = """
# 编码和规范
- **Python规范：** 严格遵循Python最佳实践（PEP 8）。
- **封装：** 尽量使用函数进行代码封装，除非用户明确要求（例如代码片段过短）。
- **变量：** 除非用户要求，避免引入新的全局变量。使用清晰、描述性的变量名。
- **可视化：** 优先使用 `matplotlib` 或 `seaborn` 绘制图表，图表应清晰易读。
- **执行方式：** 代码应直接执行，无需使用 `if __name__ == '__main__'`。
- **效率：** 确保代码简洁、高效、易于维护。
""".strip()

# For text generation modes (explain)
MARKDOWN_FORMAT = """
# 强制输出要求 (CRITICAL)
- **输出必须使用标准Markdown格式**（标题、列表、代码引用、粗体等）。
- **必须**使用代码块（```python）引用代码片段或变量。
- 文档内容必须全面、准确，并保持专业的分析口吻。
- 解释文档必须是完整的、独立的文本，无需代码执行。
""".strip()

# For text generation modes (explain)
MARKDOWN_STYLE = """
# 编码和规范
- **Python规范：** 引用代码片段时，应遵循Python最佳实践（PEP 8）。
- **变量：** 对代码中使用的核心变量和参数进行清晰的解释。
- **可视化：** 如果代码包含绘图逻辑，应解释绘图的目的和使用的库。
- **效率：** 解释中应体现对代码效率的考量。
""".strip()

# ==========================================
# 2. Mode Specific Instructions (模式特定指令)
# ==========================================

# --- CREATE Mode ---
CREATE_TASK = _load_prompt_template(
    'create_prompt.txt',
    fallback="""
# 核心任务
根据用户意图，编写新的Python代码，或在现有代码基础上完善功能。
"""
)

# --- FIX Mode ---
FIX_TASK = _load_prompt_template(
    'fix_prompt.txt',
    fallback="""
# 核心任务
修复现有代码中的错误（BUG），确保代码能正常运行。
"""
)

# --- REFACTOR Mode ---
REFACTOR_TASK = _load_prompt_template(
    'refactor_prompt.txt',
    fallback="""
# 核心任务
优化代码结构或性能，**严禁改变代码原有的功能和业务逻辑**。
"""
)

# --- EXPLAIN Mode ---
EXPLAIN_TASK = _load_prompt_template(
    'explain_prompt.txt',
    fallback="""
# 核心任务
为代码生成清晰的解释文档（Markdown格式），用于教学或文档记录。
"""
)

# --- NORMALIZE Mode ---
NORMALIZE_TASK = _load_prompt_template(
    'normalize_prompt.txt',
    fallback="""
# 核心任务
将用户提供的非标准代码重构为符合《算法节点开发指南》的标准算法函数。
请参考相关规范文档进行代码生成。
"""
)

# ==========================================
# 3. Assembly (组装)
# ==========================================

def _assemble_prompt(role, task, output_format, style):
    return f"\n{role}\n\n{COMMON_INPUT_FORMAT}\n\n{task}\n\n{output_format}\n\n{style}\n"

CREATE_SYSTEM_PROMPT = _assemble_prompt(COMMON_ROLE, CREATE_TASK, CODE_FORMAT, CODE_STYLE)
FIX_SYSTEM_PROMPT = _assemble_prompt(COMMON_ROLE, FIX_TASK, CODE_FORMAT, CODE_STYLE)
REFACTOR_SYSTEM_PROMPT = _assemble_prompt(COMMON_ROLE, REFACTOR_TASK, CODE_FORMAT, CODE_STYLE)
EXPLAIN_SYSTEM_PROMPT = _assemble_prompt(COMMON_ROLE, EXPLAIN_TASK, MARKDOWN_FORMAT, MARKDOWN_STYLE)
NORMALIZE_SYSTEM_PROMPT = _assemble_prompt(COMMON_ROLE, NORMALIZE_TASK, CODE_FORMAT, CODE_STYLE)

# Mode mapping
MODE_PROMPTS = {
    "create": CREATE_SYSTEM_PROMPT,
    "fix": FIX_SYSTEM_PROMPT,
    "refactor": REFACTOR_SYSTEM_PROMPT,
    "explain": EXPLAIN_SYSTEM_PROMPT,
    "normalize": NORMALIZE_SYSTEM_PROMPT
}

