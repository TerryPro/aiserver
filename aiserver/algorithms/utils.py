import inspect
import re
import ast
from typing import List, Dict, Any, Optional, Callable
from .base import AlgorithmParameter

def extract_imports_from_source(source: str) -> List[str]:
    """Extract import statements from source code string."""
    try:
        imports = []
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append(f"import {name.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for name in node.names:
                    imports.append(f"from {module} import {name.name}")
        return imports
    except Exception as e:
        # print(f"Warning: Failed to extract imports from source: {e}")
        return []

def extract_imports_from_func(func: Callable) -> List[str]:
    """Extract import statements from function source code."""
    try:
        source = inspect.getsource(func)
        return extract_imports_from_source(source)
    except Exception as e:
        print(f"Warning: Failed to extract imports from {func.__name__}: {e}")
        return []

def parse_docstring_params(docstring: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse parameter descriptions and metadata from docstring.
    Supports format:
    Parameters:
    param_name (type): Description
        label: Label Name
        widget: widget-type
        priority: critical
    """
    if not docstring:
        return {}
    
    params = {}
    lines = docstring.split('\n')
    in_params_section = False
    current_param = None
    param_indent = None
    
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            continue
            
        # Calculate indent
        indent = len(line) - len(line.lstrip())
        
        if stripped_line.startswith('Parameters:'):
            in_params_section = True
            continue
        if stripped_line.startswith('Returns:') or stripped_line.startswith('Example:'):
            in_params_section = False
            break
            
        if in_params_section:
            # Check if line matches parameter definition pattern
            match = re.match(r'^(\w+)\s*(?:\((.*)\))?\s*:\s*(.+)$', stripped_line)
            
            is_param_def = False
            if match:
                # If we haven't established param indent yet, this is likely a param def
                if param_indent is None:
                    is_param_def = True
                # If indentation is greater than established param indent, it's likely metadata/continuation
                elif indent > param_indent:
                    is_param_def = False
                # If indentation is same (or less), it's a new param
                else:
                    is_param_def = True
            
            if is_param_def:
                if param_indent is None:
                    param_indent = indent
                
                current_param = match.group(1)
                param_type_str = match.group(2)
                desc = match.group(3)
                params[current_param] = {"description": desc}
                if param_type_str:
                    params[current_param]["type"] = param_type_str
            
            elif current_param:
                # Not a param def, check for metadata
                meta_match = re.match(r'^(label|widget|priority|options|min|max|step|ignore|role)\s*:\s*(.+)$', stripped_line)
                if meta_match:
                    key = meta_match.group(1)
                    value = meta_match.group(2)
                    
                    # Type conversion
                    if key in ['min', 'max', 'step']:
                        try:
                            if '.' in value:
                                value = float(value)
                            else:
                                value = int(value)
                        except ValueError:
                            pass 
                    elif key == 'ignore':
                        value = value.strip().lower() == 'true'
                    elif key == 'options':
                        # Simple list parsing: [a, b, c] or a, b, c
                        val_str = value.strip('[]')
                        if val_str:
                            value_list = [v.strip() for v in val_str.split(',')]
                            # Try to convert elements
                            try:
                                value = [int(v) for v in value_list]
                            except ValueError:
                                try:
                                    value = [float(v) for v in value_list]
                                except ValueError:
                                    value = [v.strip("'").strip('"') for v in value_list]
                        else:
                            value = []
                            
                    params[current_param][key] = value
                else:
                    # Continuation of description
                    params[current_param]["description"] += " " + stripped_line
                
    return params

def infer_widget_type(name: str, param_type: str, options: List[Any] = None) -> str:
    """Infer widget type from parameter name and type."""
    if options:
        return "select"
    
    name_lower = name.lower()
    
    if "filepath" in name_lower or "file_path" in name_lower:
        return "file-selector"
    if "columns" in name_lower or "column" in name_lower:
        # Usually implies selecting columns from dataframe
        if param_type == "list":
            return "column-selector" # Multi-select
        return "column-selector" # Single select if type is str, but widget name is same
    if "color" in name_lower:
        return "color-picker"
    
    if param_type == "bool":
        return "checkbox"
    if param_type == "int":
        return "input-number"
    if param_type == "float":
        return "input-number"
        
    return "input-text"

def extract_parameters_from_func(func: Callable, overrides: Dict[str, Dict[str, Any]] = None) -> List[AlgorithmParameter]:
    """
    Extract AlgorithmParameter list from a function's signature and docstring.
    
    Args:
        func: The function to inspect.
        overrides: Dictionary mapping parameter names to attribute overrides 
                   (e.g. {'filepath': {'widget': 'file-selector', 'label': 'File Path'}}).
    """
    if overrides is None:
        overrides = {}
        
    sig = inspect.signature(func)
    doc_params = parse_docstring_params(inspect.getdoc(func))
    
    parameters = []
    
    for name, param in sig.parameters.items():
        # No longer skipping system parameters
        # if name in ['output_var', 'df']:
        #    continue
            
        # Get type info
        param_type = "str" # Default
        if param.annotation != inspect.Parameter.empty:
            if param.annotation == int:
                param_type = "int"
            elif param.annotation == float:
                param_type = "float"
            elif param.annotation == bool:
                param_type = "bool"
            elif param.annotation == list:
                param_type = "list"
            elif hasattr(param.annotation, '__name__'):
                param_type = param.annotation.__name__
        elif param.default != inspect.Parameter.empty and param.default is not None:
             if isinstance(param.default, int):
                 param_type = "int"
             elif isinstance(param.default, float):
                 param_type = "float"
             elif isinstance(param.default, bool):
                 param_type = "bool"
             elif isinstance(param.default, list):
                 param_type = "list"

        # Get default value
        default_val = param.default if param.default != inspect.Parameter.empty else None
        
        # Get docstring info (description and metadata)
        param_info = doc_params.get(name, {})
        
        # Use type from docstring if available
        doc_type = param_info.get("type")
        if doc_type:
             doc_type_lower = doc_type.lower()
             if "list" in doc_type_lower:
                 param_type = "list"
             elif "int" in doc_type_lower:
                 param_type = "int"
             elif "float" in doc_type_lower:
                 param_type = "float"
             elif "bool" in doc_type_lower:
                 param_type = "bool"
        
        if param_info.get("ignore"):
            continue
            
        description = param_info.get("description", f"Parameter {name}")
        
        # Determine priority
        # Use metadata priority if available, else infer from default value
        doc_priority = param_info.get("priority")
        if doc_priority:
            priority = doc_priority
        else:
            priority = "critical" if default_val is None or default_val == inspect.Parameter.empty else "non-critical"
        
        # Determine role
        doc_role = param_info.get("role")
        if doc_role:
            role = doc_role
        else:
            # Infer role
            if name == 'df':
                role = 'input'
            elif name == 'output_var':
                role = 'output'
            else:
                role = 'parameter'
        
        # Handle empty default for critical params (to avoid confusing UI)
        if default_val == inspect.Parameter.empty:
            default_val = ""
            if param_type == "int": default_val = 0
            if param_type == "float": default_val = 0.0
            if param_type == "bool": default_val = False
            if param_type == "list": default_val = []

        
        # Apply overrides (Highest priority)
        override_props = overrides.get(name, {})
        
        # Merge options
        options = override_props.get("options", param_info.get("options"))

        # Infer widget
        # 1. Override
        # 2. Docstring metadata
        # 3. Inference
        widget = override_props.get("widget", param_info.get("widget", infer_widget_type(name, param_type, options)))
        
        # Construct parameter
        algo_param = AlgorithmParameter(
            name=name,
            type=override_props.get("type", param_type),
            default=override_props.get("default", default_val),
            label=override_props.get("label", param_info.get("label", name.replace("_", " ").title())),
            description=override_props.get("description", description),
            widget=widget,
            options=options,
            min=override_props.get("min", param_info.get("min")),
            max=override_props.get("max", param_info.get("max")),
            step=override_props.get("step", param_info.get("step")),
            priority=override_props.get("priority", priority),
            role=override_props.get("role", role)
        )
        
        parameters.append(algo_param)
        
    return parameters

def parse_algorithm_metadata(docstring: str) -> Dict[str, Any]:
    """
    从 Docstring 解析算法元数据。
    支持格式:
    Algorithm:
        name: 算法名称
        category: 分类ID
        prompt: 提示词模板
        imports: import pandas as pd, import numpy as np
    """
    if not docstring:
        return {}
    
    metadata = {}
    lines = docstring.split('\n')
    in_algo_section = False
    
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            continue
            
        if stripped_line.startswith('Algorithm:'):
            in_algo_section = True
            continue
        
        if in_algo_section:
            # Stop if we hit another section
            if stripped_line.startswith('Parameters:') or stripped_line.startswith('Returns:') or stripped_line.startswith('Example:'):
                break
                
            # Parse key-value pairs
            match = re.match(r'^(\w+)\s*:\s*(.+)$', stripped_line)
            if match:
                key = match.group(1)
                value = match.group(2)
                
                if key == 'imports':
                    # Split imports by comma and strip
                    metadata[key] = [imp.strip() for imp in value.split(',') if imp.strip()]
                else:
                    metadata[key] = value
                    
    return metadata
