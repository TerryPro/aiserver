# -*- coding: utf-8 -*-
"""
Algorithm Validator
===================
算法代码规范验证器

功能概述:
--------
对用户编写的算法代码进行全面的规范性检查，确保代码符合项目要求，
能够被系统正确解析和使用。验证器会检查代码的语法、元数据、参数定义、
函数签名等多个方面，并返回详细的验证报告。

验证规则:
--------

1. **语法检查 (Syntax Check)**
   - 使用 Python AST 解析器检查代码语法是否正确
   - 错误级别: ERROR
   - 如果语法错误，后续检查将中止

2. **元数据解析 (Metadata Parsing)**
   - 检查是否包含 Algorithm 块（必需）
   - 验证 parse_function_code 能否成功解析代码
   - 错误级别: ERROR
   - 依赖: library.core.parse_function_code 模块

3. **必需字段检查 (Required Fields)**
   - 验证 Algorithm 块中是否包含必需字段:
     * name: 算法中文名称
     * category: 算法分类ID
   - 错误级别: ERROR

4. **分类有效性检查 (Category Validation)**
   - 验证 category 是否在预定义的有效分类列表中
   - 有效分类: load_data, data_operation, data_preprocessing, 
              eda, anomaly_detection, trend_plot, plotting
   - 错误级别: ERROR
   - 建议级别: SUGGESTION (提示有效分类列表)

5. **Docstring 规范检查 (Docstring Compliance)**
   - 检查 docstring 中是否包含独立的 import 语句（禁止）
     * import 应在 Algorithm 块的 imports 字段中声明
     * 或者在文件顶部使用标准 Python import 语句
   - 检查算法描述是否详细（至少 10 个字符）
   - 错误级别: WARNING
   - 建议级别: SUGGESTION

6. **参数定义检查 (Parameter Validation)**
   - 检查每个参数是否定义了 role 属性
     * role 必须是 'input' 或 'parameter'
   - 验证 role 的有效性
     * input: 数据输入端口（通常是 DataFrame）
     * parameter: 配置参数（用户设置的超参数）
   - 检查 priority 的有效性（如果定义）
     * critical: 重要参数，在节点面板中显示
     * non-critical: 次要参数，在属性面板中显示
     * 注意: priority 仅控制 UI 显示位置，与默认值无关
   - 检查 widget 类型的合理性
     * file-selector 不应包含硬编码的 options
   - 检查参数类型注解是否明确（非 'any'）
   - 检查参数是否有描述
   - 错误级别: ERROR (role 无效)
   - 警告级别: WARNING (缺少 role, priority 无效)
   - 建议级别: SUGGESTION (缺少类型、描述)

7. **Import 语句检查 (Import Validation)**
   - 验证 Algorithm 块中 imports 字段的格式是否正确
     * 应符合标准 Python import 语法
     * 例: import pandas as pd, from numpy import array
   - 检查代码中的 import 语句是否在文件顶部
   - 警告级别: WARNING (格式无效)
   - 建议级别: SUGGESTION (import 位置)

8. **函数签名检查 (Signature Validation)**
   - 检查函数是否有返回类型注解 (-> Type)
   - 检查所有参数是否有类型注解 (param: Type)
   - 建议级别: SUGGESTION
   - 注意: 这些是建议性检查，不影响代码运行

9. **Prompt 模板检查 (Prompt Template Validation)**
   - 检查是否定义了 prompt 模板（AI 代码生成需要）
   - 如果算法有 input 角色的参数，建议使用 {VAR_NAME} 占位符
   - 警告级别: WARNING (缺少 prompt)
   - 建议级别: SUGGESTION (建议使用占位符)

10. **返回值检查 (Return Value Validation)**
    - 检查已定义的输出端口是否有明确的类型
    - 注意: 允许函数没有输出端口（如绘图函数返回 None）
    - 建议级别: SUGGESTION

验证结果:
--------
- valid: bool - 是否通过验证（无 ERROR 级别问题）
- issues: List[ValidationIssue] - 所有发现的问题
  * level: 'error' | 'warning' | 'suggestion'
  * message: 问题描述
  * line: 代码行号（可选）
  * category: 问题分类
- metadata: Dict - 解析出的算法元数据

使用示例:
--------
```python
validator = AlgorithmValidator()
result = validator.validate(code)

if result.valid:
    print("验证通过")
else:
    for error in result.get_errors():
        print(f"错误: {error.message}")
    for warning in result.get_warnings():
        print(f"警告: {warning.message}")
```
"""

import ast
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ValidationIssue:
    """验证问题"""
    level: str  # 'error', 'warning', 'suggestion'
    message: str
    line: Optional[int] = None
    category: str = 'general'


@dataclass
class ValidationResult:
    """验证结果"""
    valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'valid': self.valid,
            'issues': [
                {
                    'level': issue.level,
                    'message': issue.message,
                    'line': issue.line,
                    'category': issue.category
                }
                for issue in self.issues
            ],
            'metadata': self.metadata
        }
    
    def get_errors(self) -> List[ValidationIssue]:
        """获取所有错误"""
        return [i for i in self.issues if i.level == 'error']
    
    def get_warnings(self) -> List[ValidationIssue]:
        """获取所有警告"""
        return [i for i in self.issues if i.level == 'warning']
    
    def get_suggestions(self) -> List[ValidationIssue]:
        """获取所有建议"""
        return [i for i in self.issues if i.level == 'suggestion']


class AlgorithmValidator:
    """算法规范验证器"""
    
    # 有效的分类
    VALID_CATEGORIES = [
        'load_data', 'data_operation', 'data_preprocessing',
        'eda', 'anomaly_detection', 'trend_plot', 'plotting'
    ]
    
    # 必需字段
    REQUIRED_FIELDS = ['name', 'category']
    
    # 有效的参数角色
    VALID_ROLES = ['input', 'parameter']
    
    # 有效的优先级
    VALID_PRIORITIES = ['critical', 'non-critical']
    
    def __init__(self):
        """初始化验证器"""
        pass
    
    def validate(self, code: str) -> ValidationResult:
        """
        验证算法代码
        
        Args:
            code: 算法代码
        
        Returns:
            ValidationResult对象
        """
        issues = []
        metadata = None
        
        # 1. 语法检查
        syntax_issues = self._check_syntax(code)
        issues.extend(syntax_issues)
        
        # 如果有语法错误,直接返回
        if any(i.level == 'error' for i in syntax_issues):
            return ValidationResult(valid=False, issues=issues)
        
        # 2. 解析算法元数据
        try:
            # aiserver.__init__已经把library路径添加到sys.path，直接导入即可
            # 注意：因为library目录本身在sys.path中，所以导入是 from core import，而不是 from library.core import
            try:
                from core import parse_function_code
            except ImportError as e:
                # 如果导入失败，尝试手动添加路径
                import sys
                import os
                        
                # 计算library路径（与aiserver.__init__中的逻辑一致）
                current_dir = os.path.dirname(os.path.abspath(__file__))
                # validators -> aiserver -> aiserver -> JuServer
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
                library_path = os.path.join(project_root, 'library')
                
                # 检查library目录是否存在
                if not os.path.exists(library_path):
                    issues.append(ValidationIssue(
                        level='error',
                        message=f'library目录不存在: {library_path}',
                        category='system'
                    ))
                    return ValidationResult(valid=False, issues=issues, metadata=metadata)
                
                # 目录存在，添加到sys.path（如果还没有）
                if library_path not in sys.path:
                    sys.path.insert(0, library_path)
                
                # 再次尝试导入
                try:
                    from core import parse_function_code
                except ImportError as e2:
                    # 仍然失败，返回详细错误
                    issues.append(ValidationIssue(
                        level='error',
                        message=f'无法加载算法解析模块。library路径: {library_path}, 错误: {str(e2)}',
                        category='system'
                    ))
                    return ValidationResult(valid=False, issues=issues, metadata=metadata)
            
            metadata = parse_function_code(code)
            
            if not metadata:
                issues.append(ValidationIssue(
                    level='error',
                    message='缺少Algorithm块，无法解析算法元数据',
                    category='metadata'
                ))
                return ValidationResult(valid=False, issues=issues, metadata=metadata)
        except Exception as e:
            issues.append(ValidationIssue(
                level='error',
                message=f'元数据解析失败: {str(e)}',
                category='metadata'
            ))
            return ValidationResult(valid=False, issues=issues, metadata=metadata)
        
        # 3. 检查必需字段
        field_issues = self._check_required_fields(metadata)
        issues.extend(field_issues)
        
        # 4. 检查分类有效性
        category_issues = self._check_category(metadata)
        issues.extend(category_issues)
        
        # 5. 检查docstring规范
        docstring_issues = self._check_docstring(code, metadata)
        issues.extend(docstring_issues)
        
        # 6. 检查参数定义
        param_issues = self._check_parameters(metadata)
        issues.extend(param_issues)
        
        # 7. 检查imports
        import_issues = self._check_imports(code, metadata)
        issues.extend(import_issues)
        
        # 8. 检查函数签名
        signature_issues = self._check_signature(code, metadata)
        issues.extend(signature_issues)
        
        # 9. 检查prompt模板
        prompt_issues = self._check_prompt(metadata)
        issues.extend(prompt_issues)
        
        # 10. 检查返回值
        return_issues = self._check_returns(metadata)
        issues.extend(return_issues)
        
        # 判断是否通过验证（只要没有error级别的问题就算通过）
        has_errors = any(i.level == 'error' for i in issues)
        
        return ValidationResult(
            valid=not has_errors,
            issues=issues,
            metadata=metadata
        )
    
    def _check_syntax(self, code: str) -> List[ValidationIssue]:
        """检查语法"""
        issues = []
        try:
            ast.parse(code)
        except SyntaxError as e:
            issues.append(ValidationIssue(
                level='error',
                message=f'语法错误: {str(e)}',
                line=e.lineno,
                category='syntax'
            ))
        return issues
    
    def _check_required_fields(self, metadata: Dict[str, Any]) -> List[ValidationIssue]:
        """检查必需字段"""
        issues = []
        for field in self.REQUIRED_FIELDS:
            if not metadata.get(field):
                issues.append(ValidationIssue(
                    level='error',
                    message=f'缺少必需字段: {field}',
                    category='metadata'
                ))
        return issues
    
    def _check_category(self, metadata: Dict[str, Any]) -> List[ValidationIssue]:
        """检查分类有效性"""
        issues = []
        category = metadata.get('category')
        
        if category and category not in self.VALID_CATEGORIES:
            issues.append(ValidationIssue(
                level='error',
                message=f'无效的分类: {category}',
                category='metadata'
            ))
            issues.append(ValidationIssue(
                level='suggestion',
                message=f'有效分类: {", ".join(self.VALID_CATEGORIES)}',
                category='metadata'
            ))
        
        return issues
    
    def _check_docstring(self, code: str, metadata: Dict[str, Any]) -> List[ValidationIssue]:
        """检查docstring规范"""
        issues = []
        
        # 检查是否在docstring中有import声明（禁止）
        if self._has_import_in_docstring(code):
            issues.append(ValidationIssue(
                level='warning',
                message='docstring中不应包含import声明',
                category='docstring'
            ))
            issues.append(ValidationIssue(
                level='suggestion',
                message='import语句应该放在文件顶部，而不是docstring中',
                category='docstring'
            ))
        
        # 检查描述是否存在
        description = metadata.get('description', '')
        if not description or len(description.strip()) < 10:
            issues.append(ValidationIssue(
                level='warning',
                message='算法描述过于简短，建议添加详细的功能说明',
                category='docstring'
            ))
        
        return issues
    
    def _check_parameters(self, metadata: Dict[str, Any]) -> List[ValidationIssue]:
        """检查参数定义"""
        issues = []
        params = metadata.get('args', [])
        
        if not params:
            issues.append(ValidationIssue(
                level='warning',
                message='函数没有参数，请确认这是否符合预期',
                category='parameters'
            ))
            return issues
        
        for param in params:
            param_name = param.get('name', 'unknown')
            
            # 检查role
            role = param.get('role')
            if not role:
                issues.append(ValidationIssue(
                    level='warning',
                    message=f'参数 {param_name} 缺少role属性',
                    category='parameters'
                ))
            elif role not in self.VALID_ROLES:
                issues.append(ValidationIssue(
                    level='error',
                    message=f'参数 {param_name} 的role "{role}" 无效，有效值: {", ".join(self.VALID_ROLES)}',
                    category='parameters'
                ))
            
            # priority和默认值无关，仅用于控制UI显示位置
            # critical: 在流程图节点面板中显示
            # non-critical: 在属性面板中显示
            # parameter角色的参数应该有默认值，但这不是强制要求
            
            # 检查priority
            priority = param.get('priority')
            if priority and priority not in self.VALID_PRIORITIES:
                issues.append(ValidationIssue(
                    level='warning',
                    message=f'参数 {param_name} 的priority "{priority}" 无效，有效值: {", ".join(self.VALID_PRIORITIES)}',
                    category='parameters'
                ))
            
            # 检查widget类型
            widget = param.get('widget')
            if widget == 'file-selector' and param.get('options'):
                issues.append(ValidationIssue(
                    level='suggestion',
                    message=f'参数 {param_name} 的file-selector选项应在前端动态获取，不应在docstring中硬编码',
                    category='parameters'
                ))
            
            # 检查类型注解
            param_type = param.get('type')
            if not param_type or param_type == 'any':
                issues.append(ValidationIssue(
                    level='suggestion',
                    message=f'参数 {param_name} 缺少明确的类型注解',
                    category='parameters'
                ))
            
            # 检查描述
            description = param.get('description', '')
            if not description:
                issues.append(ValidationIssue(
                    level='suggestion',
                    message=f'参数 {param_name} 缺少描述',
                    category='parameters'
                ))
        
        return issues
    
    def _check_imports(self, code: str, metadata: Dict[str, Any]) -> List[ValidationIssue]:
        """检查imports"""
        issues = []
        imports = metadata.get('imports', [])
        
        if imports:
            for imp in imports:
                if not self._is_valid_import(imp):
                    issues.append(ValidationIssue(
                        level='warning',
                        message=f'无效的import格式: {imp}',
                        category='imports'
                    ))
        
        # 检查import语句是否在文件顶部
        try:
            tree = ast.parse(code)
            import_found = False
            non_import_found = False
            
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    if non_import_found:
                        issues.append(ValidationIssue(
                            level='suggestion',
                            message='建议将所有import语句放在文件顶部',
                            line=node.lineno,
                            category='imports'
                        ))
                    import_found = True
                elif isinstance(node, ast.FunctionDef):
                    non_import_found = True
        except:
            pass
        
        return issues
    
    def _check_signature(self, code: str, metadata: Dict[str, Any]) -> List[ValidationIssue]:
        """检查函数签名"""
        issues = []
        
        try:
            tree = ast.parse(code)
            func_node = None
            
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    func_node = node
                    break
            
            if not func_node:
                return issues
            
            # 检查返回类型注解
            if not func_node.returns:
                issues.append(ValidationIssue(
                    level='suggestion',
                    message='函数缺少返回类型注解，建议添加 -> 返回类型',
                    category='signature'
                ))
            
            # 检查参数类型注解
            for arg in func_node.args.args:
                if not arg.annotation:
                    issues.append(ValidationIssue(
                        level='suggestion',
                        message=f'参数 {arg.arg} 缺少类型注解',
                        category='signature'
                    ))
        except:
            pass
        
        return issues
    
    def _check_prompt(self, metadata: Dict[str, Any]) -> List[ValidationIssue]:
        """检查prompt模板"""
        issues = []
        prompt = metadata.get('prompt', '')
        
        if not prompt:
            issues.append(ValidationIssue(
                level='warning',
                message='缺少prompt模板，这会影响AI代码生成功能',
                category='prompt'
            ))
        else:
            # 检查是否使用了占位符
            if '{VAR_NAME}' not in prompt and '{{VAR_NAME}}' not in prompt:
                # 检查是否有input角色的参数
                has_input = any(p.get('role') == 'input' for p in metadata.get('args', []))
                if has_input:
                    issues.append(ValidationIssue(
                        level='suggestion',
                        message='prompt模板建议使用 {{VAR_NAME}} 占位符引用输入变量',
                        category='prompt'
                    ))
        
        return issues
    
    def _check_returns(self, metadata: Dict[str, Any]) -> List[ValidationIssue]:
        """检查返回值"""
        issues = []
        outputs = metadata.get('outputs', [])
        
        # 如果没有输出端口，不发出警告（允许返回None的函数，如绘图函数）
        # 只有当明确有返回值但未定义时才提示
        # 这个检查可以在后续版本中增强，通过AST分析函数体中的return语句
        
        if outputs:
            for output in outputs:
                output_name = output.get('name', 'unknown')
                output_type = output.get('type')
                
                if not output_type or output_type == 'any':
                    issues.append(ValidationIssue(
                        level='suggestion',
                        message=f'返回值 {output_name} 缺少明确的类型定义',
                        category='returns'
                    ))
        
        return issues
    
    def _has_import_in_docstring(self, code: str) -> bool:
        """检查docstring中是否有import声明"""
        try:
            tree = ast.parse(code)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    docstring = ast.get_docstring(node)
                    if docstring and 'import' in docstring.lower():
                        # 检查是否在imports字段外的其他地方有import
                        # 排除Algorithm块中的imports字段
                        lines = docstring.split('\n')
                        in_algorithm_block = False
                        
                        for line in lines:
                            if 'Algorithm:' in line:
                                in_algorithm_block = True
                            elif line.strip() and not line.startswith(' '):
                                in_algorithm_block = False
                            
                            # 检查是否有独立的import语句（不在imports字段中）
                            if not in_algorithm_block or 'imports:' not in line:
                                if re.search(r'^\s*import\s+', line, re.IGNORECASE):
                                    return True
                                if re.search(r'^\s*from\s+\w+\s+import\s+', line, re.IGNORECASE):
                                    return True
        except:
            pass
        
        return False
    
    def _is_valid_import(self, imp: str) -> bool:
        """检查import语句格式是否有效"""
        patterns = [
            r'^import\s+[\w.]+(\s+as\s+\w+)?$',
            r'^from\s+[\w.]+\s+import\s+[\w,\s]+(\s+as\s+\w+)?$'
        ]
        return any(re.match(p, imp.strip()) for p in patterns)
