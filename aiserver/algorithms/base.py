# -*- coding: utf-8 -*-
"""
Algorithm Base Types (Compatibility Layer)
==========================================
保持与现有代码的兼容性，实际实现已迁移到 library/core 和 adapter.py
"""

# 从核心模块重新导出类型（保持后向兼容）
from core.models import (
    AlgorithmPort as Port,
    AlgorithmParameter,
    AlgorithmMetadata
)

# 从适配器导入Algorithm和AlgorithmCategory
from .adapter import Algorithm, AlgorithmCategory

# 导出所有类型（保持兼容）
__all__ = [
    'Port',
    'AlgorithmParameter', 
    'AlgorithmMetadata',
    'Algorithm',
    'AlgorithmCategory'
]
