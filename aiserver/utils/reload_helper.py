# -*- coding: utf-8 -*-
"""
Module Reload Helper
====================
算法模块重载工具
"""

import sys
import importlib
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


def reload_algorithm_modules() -> Dict[str, Any]:
    """
    重载算法相关的所有模块
    
    包括：
    - algorithm.* 子模块
    - algorithm 主模块
    - aiserver.algorithms
    - aiserver.lib.library
    - aiserver.prompts.algorithm
    
    Returns:
        包含重载统计信息的字典：
        {
            "reloaded_submodules": int,  # 重载的子模块数量
            "reloaded_main_modules": list,  # 重载的主模块列表
            "errors": list  # 重载失败的模块列表
        }
    """
    result = {
        "reloaded_submodules": 0,
        "reloaded_main_modules": [],
        "errors": []
    }
    
    # 清除导入缓存
    importlib.invalidate_caches()
    
    # 1. 重载 algorithm.* 子模块
    for name in list(sys.modules.keys()):
        if name.startswith('algorithm.') and sys.modules[name]:
            try:
                importlib.reload(sys.modules[name])
                result["reloaded_submodules"] += 1
            except Exception as e:
                logger.warning(f"Failed to reload {name}: {e}")
                result["errors"].append({"module": name, "error": str(e)})
    
    # 2. 重载主模块（顺序很重要）
    main_modules = [
        'algorithm',
        'aiserver.algorithms',
        'aiserver.lib.library',
        'aiserver.prompts.algorithm'
    ]
    
    for mod_name in main_modules:
        if mod_name in sys.modules:
            try:
                importlib.reload(sys.modules[mod_name])
                result["reloaded_main_modules"].append(mod_name)
                logger.info(f"Reloaded {mod_name}")
            except Exception as e:
                logger.error(f"Failed to reload {mod_name}: {e}")
                result["errors"].append({"module": mod_name, "error": str(e)})
    
    logger.info(
        f"Reload complete: {result['reloaded_submodules']} submodules, "
        f"{len(result['reloaded_main_modules'])} main modules"
    )
    
    return result
