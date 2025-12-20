# -*- coding: utf-8 -*-
"""
Algorithm Code Manager
======================
算法文件的CRUD操作以及代码解析/生成

使用 library/core 核心模块进行解析和生成
"""

import os
import sys
import ast
import logging

# 从Python源码中提取算法元数据（使用核心CodeParser）
from core import parse_function_code

logger = logging.getLogger(__name__)

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../library/algorithm"))

def get_algorithm_path(algo_id: str) -> str:
    """Find the file path for a given algorithm ID."""
    # Search in all subdirectories
    for category in os.listdir(BASE_PATH):
        category_path = os.path.join(BASE_PATH, category)
        if os.path.isdir(category_path):
            file_path = os.path.join(category_path, f"{algo_id}.py")
            if os.path.exists(file_path):
                return file_path
    return None

def update_init_file(category_path: str):
    """Update __init__.py in the category directory to import all algorithms."""
    init_path = os.path.join(category_path, "__init__.py")
    files = sorted([f for f in os.listdir(category_path) if f.endswith(".py") and f != "__init__.py"])
    
    lines = []
    for f in files:
        mod_name = f[:-3]
        lines.append(f"from .{mod_name} import {mod_name}")
        
    with open(init_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")

def add_function(category: str, code: str) -> bool:
    """Add a new algorithm function."""
    # 1. Parse code to find function name (algorithm ID)
    try:
        tree = ast.parse(code)
        func_name = None
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                func_name = node.name
                break
        
        if not func_name:
            raise ValueError("No function definition found in provided code")
            
        # 2. Check if already exists
        if get_algorithm_path(func_name):
            raise ValueError(f"Algorithm {func_name} already exists")
            
        # 3. Create category directory if not exists
        category_path = os.path.join(BASE_PATH, category)
        if not os.path.exists(category_path):
            os.makedirs(category_path)
            
        # 4. Write file
        file_path = os.path.join(category_path, f"{func_name}.py")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)
            
        # 5. Update __init__.py
        update_init_file(category_path)
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to add function: {e}")
        raise e

def update_function(algo_id: str, code: str) -> bool:
    """Update an existing algorithm function."""
    file_path = get_algorithm_path(algo_id)
    if not file_path:
        raise ValueError(f"Algorithm {algo_id} not found")
        
    # Verify function name matches filename (optional but good for consistency)
    try:
        tree = ast.parse(code)
        func_name = None
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                func_name = node.name
                break
        
        if func_name and func_name != algo_id:
            # If name changed, we might need to rename the file?
            # For now, let's enforce ID = func_name
            # Or just warn?
            # If user renames function, we should treat it as delete + add?
            # Or rename the file?
            # Let's assume user keeps the name, or if they change it, we rename the file.
            
            # Rename logic:
            new_file_path = os.path.join(os.path.dirname(file_path), f"{func_name}.py")
            if os.path.exists(new_file_path):
                raise ValueError(f"Target algorithm {func_name} already exists")
            
            # Delete old, write new
            os.remove(file_path)
            file_path = new_file_path
            
    except Exception as e:
        logger.error(f"Failed to parse code: {e}")
        raise e
        
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)
            
        # Update init just in case
        update_init_file(os.path.dirname(file_path))
        return True
    except Exception as e:
        logger.error(f"Failed to update function: {e}")
        raise e

def delete_function(algo_id: str) -> bool:
    """Delete an algorithm function."""
    file_path = get_algorithm_path(algo_id)
    if not file_path:
        raise ValueError(f"Algorithm {algo_id} not found")
        
    try:
        os.remove(file_path)
        update_init_file(os.path.dirname(file_path))
        return True
    except Exception as e:
        logger.error(f"Failed to delete function: {e}")
        raise e

def get_function_code(algo_id: str) -> str:
    """Get the source code of an algorithm function."""
    file_path = get_algorithm_path(algo_id)
    if not file_path:
        return None
        
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to read function code: {e}")
        return None

def generate_function_code(metadata: dict, existing_code: str = None) -> str:
    """
    Generate Python code for an algorithm function based on metadata.
    If existing_code is provided, preserves the function body and imports.
    
    使用 core/generator 模块实现，避免代码重复
    """
    from core import AlgorithmMetadata, CodeGenerator
    from core.extractor import CodeExtractor
    
    # 将字典转换为 AlgorithmMetadata 对象
    algo_metadata = AlgorithmMetadata.from_dict(metadata)
    
    # 提取现有函数体（如果有）
    existing_body = None
    if existing_code:
        extractor = CodeExtractor()
        existing_body = extractor.extract_function_body(existing_code)
    
    # 使用核心生成器 - 传入existing_code以保留imports
    generator = CodeGenerator()
    return generator.generate_function_code(algo_metadata, existing_body, existing_code)

def parse_function_code(code: str) -> dict:
    """
    Parse Python code to extract algorithm metadata.
    
    使用 core/parser 模块实现，避免代码重复
    """
    from core import parse_function_code as core_parse
    return core_parse(code)
