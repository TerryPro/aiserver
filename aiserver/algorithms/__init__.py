# -*- coding: utf-8 -*-
"""
Algorithm Module
================
使用 library/core 核心模块进行算法扫描和管理

保持与现有代码的API兼容性
"""

from typing import List, Dict, Any
import algorithm

# 从c核心模块导入（路径自动初始化）
from core import (
    AlgorithmMetadata,
    AlgorithmParameter,
    AlgorithmPort as Port,
    LibraryScanner,
    get_category_labels
)

# 从适配器导入兼容类型
from .adapter import (
    Algorithm,
    AlgorithmCategory,
    scan_and_create_algorithms,
    build_algorithm_prompts,
    build_algorithm_dicts
)

# Import CATEGORY_LABELS
try:
    from algorithm import CATEGORY_LABELS
except ImportError:
    print("Warning: CATEGORY_LABELS not found in algorithm package")
    CATEGORY_LABELS = get_category_labels()

# 使用核心扫描器扫描算法
all_algorithms: List[Algorithm] = scan_and_create_algorithms(algorithm)
algorithms_dict = {algo.id: algo for algo in all_algorithms}

# 构建兼容的字典结构
ALGORITHM_PARAMETERS, ALGORITHM_TEMPLATES, ALGORITHM_IMPORTS = build_algorithm_dicts(all_algorithms)

# 构建算法提示
ALGORITHM_PROMPTS = build_algorithm_prompts(all_algorithms)
