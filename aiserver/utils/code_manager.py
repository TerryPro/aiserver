import os
import ast
import logging
from ..algorithms.utils import parse_algorithm_metadata, parse_docstring_params, extract_imports_from_source

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
    If existing_code is provided, preserves the function body.
    """
    func_name = metadata.get('id', 'new_algorithm')
    name = metadata.get('name', 'Algorithm Name')
    category = metadata.get('category', 'uncategorized')
    description = metadata.get('description', 'Description')
    prompt = metadata.get('prompt', f"Perform {name} on {{VAR_NAME}}")
    imports = metadata.get('imports', [])
    
    args = metadata.get('args', [])
    inputs = metadata.get('inputs', [])
    outputs = metadata.get('outputs', [])
    
    # Merge imports from existing code if available
    if existing_code:
        try:
            existing_imports = extract_imports_from_source(existing_code)
            for imp in existing_imports:
                if imp not in imports:
                    imports.append(imp)
        except Exception as e:
            logger.warning(f"Failed to extract imports from existing code: {e}")

    # Construct function signature
    sig_args = []
    
    # Add inputs (typically DataFrames)
    for inp in inputs:
        # Default to pd.DataFrame type hint if not specified
        t = inp.get('type', 'pd.DataFrame')
        sig_args.append(f"{inp['name']}: {t}")
        
    # Add args
    # Separate required and optional args to ensure valid Python syntax
    # (non-default args cannot follow default args)
    required_args = []
    optional_args = []
    
    for arg in args:
        if arg.get('default') is not None and arg.get('default') != "":
             optional_args.append(arg)
        else:
             required_args.append(arg)
             
    # Sort: Required args first, then optional args
    sorted_args = required_args + optional_args
    
    for arg in sorted_args:
        a_name = arg['name']
        a_type = arg.get('type', 'any')
        default = arg.get('default')
        
        # Format default value
        if default is not None and default != "":
            if isinstance(default, str):
                sig_args.append(f"{a_name}: {a_type} = '{default}'")
            else:
                sig_args.append(f"{a_name}: {a_type} = {default}")
        else:
            sig_args.append(f"{a_name}: {a_type}")
            
    sig_str = ", ".join(sig_args)
    
    # Imports string
    # Ensure we have essential imports
    if imports:
        # Check if pandas is needed (usually yes for algorithms)
        if not any("pandas" in i for i in imports):
             imports = ["import pandas as pd"] + imports
    else:
        imports = ["import pandas as pd"]

    imports_str = "\n".join(imports)
    
    # Docstring construction
    docstring_lines = []
    docstring_lines.append(f'{description}')
    docstring_lines.append('')
    docstring_lines.append('Algorithm:')
    docstring_lines.append(f'    name: {name}')
    docstring_lines.append(f'    category: {category}')
    docstring_lines.append(f'    prompt: {prompt}')
    
    clean_imports = [i.replace('\n', '') for i in imports]
    if clean_imports:
        docstring_lines.append(f'    imports: {", ".join(clean_imports)}')
        
    docstring_lines.append('')
    docstring_lines.append('Parameters:')
    
    # Merge inputs and args for Docstring Parameters section
    # Inputs usually have role: input
    for inp in inputs:
        p_name = inp['name']
        p_type = inp.get('type', 'pd.DataFrame')
        docstring_lines.append(f'{p_name} ({p_type}): Input DataFrame.')
        docstring_lines.append(f'    role: input')

    for arg in args:
        p_name = arg['name']
        p_type = arg.get('type', 'any')
        p_desc = arg.get('description', f'Parameter {p_name}')
        
        docstring_lines.append(f'{p_name} ({p_type}): {p_desc}.')
        
        # Add metadata
        if arg.get('label'): docstring_lines.append(f'    label: {arg["label"]}')
        if arg.get('widget'): docstring_lines.append(f'    widget: {arg["widget"]}')
        if arg.get('options'): 
             # Format options as JSON-like list
             import json
             opts_str = json.dumps(arg['options'])
             docstring_lines.append(f'    options: {opts_str}')
             
        if arg.get('min') is not None: docstring_lines.append(f'    min: {arg["min"]}')
        if arg.get('max') is not None: docstring_lines.append(f'    max: {arg["max"]}')
        if arg.get('step') is not None: docstring_lines.append(f'    step: {arg["step"]}')
        if arg.get('priority'): docstring_lines.append(f'    priority: {arg["priority"]}')
        if arg.get('role'): docstring_lines.append(f'    role: {arg["role"]}')
        
    docstring_lines.append('')
    docstring_lines.append('Returns:')
    
    if not outputs:
        docstring_lines.append('pd.DataFrame: Result DataFrame.')
    elif len(outputs) == 1:
        docstring_lines.append(f'{outputs[0].get("type", "pd.DataFrame")}: Result.')
    else:
        types = [o.get("type", "pd.DataFrame") for o in outputs]
        docstring_lines.append(f'Tuple[{", ".join(types)}]: Results.')
        
    docstring_str = "\n    ".join(docstring_lines)
    
    # Extract existing body if available
    body_str = f"    # Implementation\n    return {inputs[0]['name'] if inputs else 'pd.DataFrame()'}"
    
    if existing_code:
        try:
            tree = ast.parse(existing_code)
            old_func = None
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    old_func = node
                    break
            
            if old_func:
                # Get body source
                # We need to extract the lines corresponding to the body
                # ast doesn't give direct source, so we might need to be careful
                # Alternative: Use the original string and slicing
                # But existing_code might have comments etc.
                
                # Simple approach: Identify the function body start
                # The body starts after the docstring (if any) or after the signature
                
                # Let's rely on ast.get_source_segment if available (Python 3.8+)
                # Or just use the logic from frontend: Find docstring end and take everything after
                
                # Let's try to locate the docstring node
                doc_node = ast.get_docstring(old_func, clean=False)
                
                if doc_node:
                     # Find the docstring in the body
                     # Since ast.get_docstring returned a value, body[0] MUST be the docstring node
                     # We can safely use it to find where the docstring ends
                     try:
                         last_doc_line = old_func.body[0].end_lineno
                         
                         lines = existing_code.splitlines()
                         # lines is 0-indexed, lineno is 1-indexed
                         body_lines = lines[last_doc_line:]
                         body_str = "\n".join(body_lines)
                     except Exception as e:
                         logger.warning(f"Error extracting body after docstring: {e}")
                         # Fallback to default behavior (keep body_str as default or try another method)
                         pass
                else:
                    # No docstring, body starts at first statement
                    first_stmt = old_func.body[0]
                    start_line = first_stmt.lineno - 1
                    lines = existing_code.splitlines()
                    body_lines = lines[start_line:]
                    # We need to preserve indentation relative to the function
                    # But if we just take the lines, they should be indented
                    body_str = "\n".join(body_lines)

        except Exception as e:
            logger.warning(f"Failed to extract body from existing code: {e}")
            # Fallback to default body

    code = f"""{imports_str}

def {func_name}({sig_str}) -> pd.DataFrame:
    \"\"\"
    {docstring_str}
    \"\"\"
{body_str}
"""
    return code

def parse_function_code(code: str) -> dict:
    """
    Parse Python code to extract algorithm metadata.
    """
    try:
        tree = ast.parse(code)
        func_node = None
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                func_node = node
                break
        
        if not func_node:
             return None
             
        docstring = ast.get_docstring(func_node)
        if not docstring:
            return None
            
        metadata = parse_algorithm_metadata(docstring)
        params_meta = parse_docstring_params(docstring)
        imports = extract_imports_from_source(code)
        
        # Extract args from signature
        args = []
        inputs = []
        
        for arg in func_node.args.args:
            name = arg.arg
            
            # Skip self/cls if any (though these are functions)
            
            p_meta = params_meta.get(name, {})
            role = p_meta.get('role')
            
            # Infer role if not set
            if not role:
                if name == 'df' or 'df_' in name:
                    role = 'input'
                else:
                    role = 'parameter'
            
            item = {
                "name": name,
                "type": p_meta.get('type', 'any'),
                "description": p_meta.get('description', ''),
                # Add other meta
                "label": p_meta.get('label'),
                "widget": p_meta.get('widget'),
                "options": p_meta.get('options'),
                "min": p_meta.get('min'),
                "max": p_meta.get('max'),
                "step": p_meta.get('step'),
                "priority": p_meta.get('priority'),
                "role": role
            }
            
            # Clean up None values
            item = {k: v for k, v in item.items() if v is not None}
            
            if role == 'input':
                inputs.append(item)
            else:
                args.append(item)
                
        # Outputs inference (simple for now)
        outputs = []
        # Default output
        outputs.append({"name": "result", "type": "pd.DataFrame"})
            
        return {
             "id": func_node.name,
             "category": metadata.get('category'),
             "name": metadata.get('name'),
             "description": docstring.split('\\n')[0].strip(),
             "prompt": metadata.get('prompt'),
             "imports": imports,
             "args": args,
             "inputs": inputs,
             "outputs": outputs,
             "code": code
        }
    except Exception as e:
        logger.error(f"Failed to parse code: {e}")
        return None
