import ast
import logging

logger = logging.getLogger(__name__)

def extract_code_metadata(code: str) -> dict:
    """
    使用AST解析代码，提取全局变量、函数和类的定义信息。
    
    Args:
        code (str): Python代码字符串
        
    Returns:
        dict: 包含 'variables', 'functions', 'classes' 的字典
    """
    metadata = {
        "variables": [],
        "functions": [],
        "classes": []
    }
    
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        logger.warning(f"AST解析失败: {e}")
        return metadata

    for node in tree.body:
        # 提取变量赋值
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    metadata["variables"].append({
                        "name": target.id,
                        "line": node.lineno
                    })
        # 提取函数定义
        elif isinstance(node, ast.FunctionDef):
            metadata["functions"].append({
                "name": node.name,
                "doc": ast.get_docstring(node),
                "line": node.lineno,
                "args": [arg.arg for arg in node.args.args]
            })
        # 提取类定义
        elif isinstance(node, ast.ClassDef):
            metadata["classes"].append({
                "name": node.name,
                "doc": ast.get_docstring(node),
                "line": node.lineno
            })
            
    return metadata
