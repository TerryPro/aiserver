# -*- coding: utf-8 -*-
"""
Algorithm Save Handler
======================
处理算法保存请求的API端点
"""

import json
import logging
import re
import ast
from jupyter_server.base.handlers import APIHandler
from ..utils import code_manager
from ..utils.reload_helper import reload_algorithm_modules

logger = logging.getLogger(__name__)


class SaveAlgorithmHandler(APIHandler):
    """
    处理算法保存请求
    
    POST /aiserver/algorithm/save
    Request Body:
    {
        "code": "算法函数代码（包含标准docstring）",
        "category": "算法分类（可选，将从docstring解析）",
        "overwrite": false
    }
    
    Response:
    {
        "success": true,
        "algorithm_id": "function_name",
        "file_path": "library/algorithm/category/function_name.py",
        "message": "算法保存成功"
    }
    """
    
    def post(self):
        try:
            data = json.loads(self.request.body)
            
            code = data.get("code")
            category = data.get("category")
            overwrite = data.get("overwrite", False)
            
            if not code:
                self.set_status(400)
                self.finish(json.dumps({"error": "缺少代码内容"}))
                return
            
            # 提取函数名（算法ID）
            algorithm_id = self._extract_function_name(code)
            if not algorithm_id:
                self.set_status(400)
                self.finish(json.dumps({"error": "代码中未找到函数定义"}))
                return
            
            # 如果未提供分类，尝试从docstring解析
            if not category:
                category = self._extract_category_from_docstring(code)
                if not category:
                    category = "data_operation"  # 默认分类
                    logger.warning(f"未找到分类信息，使用默认分类: {category}")
            
            # 检查是否已存在
            existing_path = code_manager.get_algorithm_path(algorithm_id)
            if existing_path and not overwrite:
                self.set_status(409)  # Conflict
                self.finish(json.dumps({
                    "error": f"算法 {algorithm_id} 已存在",
                    "algorithm_id": algorithm_id,
                    "existing_path": existing_path,
                    "suggestion": "请设置 overwrite=true 以覆盖现有算法"
                }))
                return
            
            # 保存或更新算法
            if existing_path:
                # 覆盖现有算法
                code_manager.update_function(algorithm_id, code)
                action = "更新"
            else:
                # 添加新算法
                code_manager.add_function(category, code)
                action = "保存"
            
            # 触发热更新
            reload_algorithm_modules()
            
            # 返回成功结果
            file_path = code_manager.get_algorithm_path(algorithm_id)
            self.finish(json.dumps({
                "success": True,
                "algorithm_id": algorithm_id,
                "file_path": file_path,
                "category": category,
                "message": f"算法{action}成功"
            }, ensure_ascii=False))
            
        except ValueError as e:
            logger.error(f"保存算法失败（参数错误）: {str(e)}")
            self.set_status(400)
            self.finish(json.dumps({"error": str(e)}))
        except Exception as e:
            logger.error(f"保存算法失败: {str(e)}", exc_info=True)
            self.set_status(500)
            self.finish(json.dumps({"error": f"服务器错误: {str(e)}"}))
    
    def _extract_function_name(self, code: str) -> str:
        """从代码中提取函数名"""
        try:
            tree = ast.parse(code)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    return node.name
        except Exception as e:
            logger.error(f"解析函数名失败: {e}")
        return None
    
    def _extract_category_from_docstring(self, code: str) -> str:
        """从docstring的Algorithm块中提取category"""
        try:
            # 提取docstring
            tree = ast.parse(code)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    docstring = ast.get_docstring(node)
                    if docstring:
                        # 使用正则提取 category
                        match = re.search(r'category:\s*(\w+)', docstring)
                        if match:
                            return match.group(1)
        except Exception as e:
            logger.error(f"从docstring提取category失败: {e}")
        return None
