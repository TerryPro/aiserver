# -*- coding: utf-8 -*-
"""
Algorithm Validate Handler
===========================
提供算法代码验证的API端点
"""

import json
import logging
from jupyter_server.base.handlers import APIHandler
from ..validators import AlgorithmValidator

logger = logging.getLogger(__name__)


class ValidateAlgorithmHandler(APIHandler):
    """
    处理算法验证请求
    
    POST /aiserver/algorithm/validate
    Request Body:
    {
        "code": "算法函数代码（包含标准docstring）"
    }
    
    Response:
    {
        "valid": true/false,
        "issues": [
            {
                "level": "error" | "warning" | "suggestion",
                "message": "问题描述",
                "line": 10,  // 可选
                "category": "metadata" | "parameters" | "imports" | ...
            }
        ],
        "metadata": {
            "id": "function_name",
            "name": "算法名称",
            "category": "分类",
            ...
        },
        "summary": {
            "errors": 0,
            "warnings": 2,
            "suggestions": 5
        }
    }
    """
    
    def check_xsrf_cookie(self):
        return

    def post(self):
        try:
            data = json.loads(self.request.body)
            code = data.get("code")
            
            if not code:
                self.set_status(400)
                self.finish(json.dumps({"error": "缺少代码内容"}))
                return
            
            # 执行验证
            validator = AlgorithmValidator()
            validation_result = validator.validate(code)
            
            # 构建响应
            result = validation_result.to_dict()
            
            # 添加统计摘要
            result["summary"] = {
                "errors": len(validation_result.get_errors()),
                "warnings": len(validation_result.get_warnings()),
                "suggestions": len(validation_result.get_suggestions())
            }
            
            self.finish(json.dumps(result, ensure_ascii=False))
            
        except Exception as e:
            logger.error(f"验证算法失败: {str(e)}", exc_info=True)
            self.set_status(500)
            self.finish(json.dumps({"error": f"服务器错误: {str(e)}"}))
