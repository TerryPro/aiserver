# -*- coding: utf-8 -*-
"""
Algorithm Adapter
=================
适配 library/core 模块到 aiserver 的兼容层

保持与现有 aiserver.algorithms 模块的API兼容性
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

# 从核心模块导入
from core import (
    AlgorithmMetadata as CoreAlgorithmMetadata,
    AlgorithmParameter as CoreAlgorithmParameter,
    AlgorithmPort as CoreAlgorithmPort,
    LibraryScanner,
    get_category_labels
)


# 重新导出核心类型（保持兼容性）
AlgorithmParameter = CoreAlgorithmParameter
Port = CoreAlgorithmPort


@dataclass
class Algorithm:
    """
    算法类（兼容现有API）
    
    包装 CoreAlgorithmMetadata 以保持与现有代码的兼容性
    """
    id: str
    name: str
    category: str
    prompt: str
    description: str = ""
    template: str = ""
    parameters: List[AlgorithmParameter] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    inputs: List[Port] = field(default_factory=list)
    outputs: List[Port] = field(default_factory=list)
    
    # 内部使用，存储核心元数据
    _core_metadata: Optional[CoreAlgorithmMetadata] = field(default=None, repr=False)
    
    def initialize(self):
        """初始化方法（保持兼容，核心模块已在扫描时完成初始化）"""
        pass
    
    def to_prompt_dict(self) -> Dict[str, Any]:
        """转换为AI提示格式"""
        return {
            "id": self.id,
            "name": self.name,
            "prompt": self.prompt
        }
    
    def to_port_dict(self) -> Dict[str, List[Dict[str, str]]]:
        """转换为端口信息格式"""
        return {
            "inputs": [{"name": p.name, "type": p.type} for p in self.inputs],
            "outputs": [{"name": p.name, "type": p.type} for p in self.outputs]
        }
    
    @classmethod
    def from_core_metadata(cls, metadata: CoreAlgorithmMetadata) -> 'Algorithm':
        """从核心元数据创建Algorithm实例"""
        return cls(
            id=metadata.id,
            name=metadata.name,
            category=metadata.category,
            prompt=metadata.prompt,
            description=metadata.description,
            template=metadata.template,
            parameters=metadata.parameters,
            imports=metadata.imports,
            inputs=metadata.inputs,
            outputs=metadata.outputs,
            _core_metadata=metadata
        )


@dataclass
class AlgorithmCategory:
    """算法分类（保持兼容）"""
    id: str
    label: str
    algorithms: List[Algorithm] = field(default_factory=list)

    def add_algorithm(self, algo: Algorithm):
        self.algorithms.append(algo)

    def to_prompt_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "algorithms": [algo.to_prompt_dict() for algo in self.algorithms]
        }


def scan_and_create_algorithms(package) -> List[Algorithm]:
    """
    扫描包并创建Algorithm实例列表
    
    Args:
        package: 要扫描的Python包
        
    Returns:
        Algorithm实例列表
    """
    scanner = LibraryScanner(package)
    metadata_by_category = scanner.scan()
    
    algorithms = []
    for cat_id, metadata_list in metadata_by_category.items():
        for metadata in metadata_list:
            algo = Algorithm.from_core_metadata(metadata)
            algorithms.append(algo)
    
    return algorithms


def build_algorithm_prompts(algorithms: List[Algorithm]) -> Dict[str, Dict[str, Any]]:
    """
    构建算法提示字典
    
    Args:
        algorithms: Algorithm实例列表
        
    Returns:
        按分类组织的算法提示字典
    """
    cat_labels = get_category_labels()
    
    prompts = {
        cat: {"label": label, "algorithms": []}
        for cat, label in cat_labels.items()
    }
    
    for algo in algorithms:
        if algo.category in prompts:
            prompts[algo.category]["algorithms"].append(algo.to_prompt_dict())
        else:
            if algo.category not in prompts:
                prompts[algo.category] = {"label": algo.category, "algorithms": []}
            prompts[algo.category]["algorithms"].append(algo.to_prompt_dict())
    
    return prompts


def build_algorithm_dicts(algorithms: List[Algorithm]) -> Dict[str, Dict[str, Any]]:
    """
    构建算法字典（用于快速查找）
    
    Returns:
        ALGORITHM_PARAMETERS, ALGORITHM_TEMPLATES, ALGORITHM_IMPORTS
    """
    parameters = {}
    templates = {}
    imports = {}
    
    for algo in algorithms:
        if algo.parameters:
            parameters[algo.id] = [p.to_dict() for p in algo.parameters]
        
        templates[algo.id] = algo.template.strip() if algo.template else ""
        
        if algo.imports:
            imports[algo.id] = algo.imports
    
    return parameters, templates, imports
