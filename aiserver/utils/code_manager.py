import os
import ast
import logging
from ..algorithms.utils import parse_algorithm_metadata, parse_docstring_params, extract_imports_from_source, parse_docstring_returns

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
    
    def normalize_type(t: str) -> str:
        if t == 'DataFrame': return 'pd.DataFrame'
        if t == 'tuple': return 'tuple'
        return t

    # Merge imports from existing code if available
    if existing_code:
        try:
            existing_imports = extract_imports_from_source(existing_code)
            
            # Build a set of imported module names to avoid duplicates
            # Extract module names from current imports for comparison
            def get_module_name(imp_str: str) -> str:
                """Extract the main module name from an import statement."""
                # For 'import pandas as pd' -> 'pandas'
                # For 'import pandas' -> 'pandas'
                # For 'from typing import List' -> 'typing'
                if imp_str.startswith('from '):
                    # from X import Y
                    parts = imp_str.split()
                    if len(parts) >= 2:
                        return parts[1]
                elif imp_str.startswith('import '):
                    # import X as Y or import X
                    parts = imp_str.replace('import ', '').split(' as ')
                    return parts[0].strip()
                return imp_str
            
            # Get module names from current imports
            current_modules = {get_module_name(imp) for imp in imports}
            
            # Add existing imports that don't duplicate module names
            for imp in existing_imports:
                module_name = get_module_name(imp)
                # Only add if this module is not already imported
                if module_name not in current_modules and imp not in imports:
                    imports.append(imp)
                    current_modules.add(module_name)
        except Exception as e:
            logger.warning(f"Failed to extract imports from existing code: {e}")

    # Construct function signature
    sig_args = []
    
    # Add inputs (typically DataFrames)
    for inp in inputs:
        # Default to pd.DataFrame type hint if not specified
        t = normalize_type(inp.get('type', 'pd.DataFrame'))
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
        a_type = normalize_type(arg.get('type', 'any'))
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
    # Check if pandas is needed (check for any pandas import)
    has_pandas = False
    for imp in imports:
        if "pandas" in imp and "import" in imp:
            has_pandas = True
            break
    
    if not has_pandas:
        # Add pandas with standard alias
        imports = ["import pandas as pd"] + imports


    # Check for typing imports
    typing_types = set()
    if len(outputs) > 0:
        typing_types.add("Optional")
    if len(outputs) > 1:
        typing_types.add("Tuple")

    # Check args and inputs for typing types
    for arg in args + inputs:
        t = arg.get('type', '')
        if 'List' in t: typing_types.add('List')
        if 'Dict' in t: typing_types.add('Dict')
        if 'Tuple' in t: typing_types.add('Tuple')
        if 'Optional' in t: typing_types.add('Optional')
        if 'Union' in t: typing_types.add('Union')
        if 'Any' in t: typing_types.add('Any')

    for t in sorted(list(typing_types)):
        # Check if already imported
        already_imported = False
        for imp in imports:
            if f"import {t}" in imp or f", {t}" in imp or f" {t}," in imp:
                already_imported = True
                break
        if not already_imported:
             imports.append(f"from typing import {t}")

    imports_str = "\n".join(imports)
    
    # Return type annotation
    if not outputs:
        ret_annotation = "-> None"
    elif len(outputs) == 1:
        t = normalize_type(outputs[0].get('type', 'pd.DataFrame'))
        ret_annotation = f"-> Optional[{t}]"
    else:
        types = [f"Optional[{normalize_type(o.get('type', 'pd.DataFrame'))}]" for o in outputs]
        ret_annotation = f"-> Tuple[{', '.join(types)}]"

    # Docstring construction
    docstring_lines = []
    docstring_lines.append(f'{description}')
    docstring_lines.append('')
    docstring_lines.append('Algorithm:')
    docstring_lines.append(f'    name: {name}')
    docstring_lines.append(f'    category: {category}')
    docstring_lines.append(f'    prompt: {prompt}')
        
    docstring_lines.append('')
    docstring_lines.append('Parameters:')
    
    # Merge inputs and args for Docstring Parameters section
    # Inputs usually have role: input
    for inp in inputs:
        p_name = inp['name']
        p_type = normalize_type(inp.get('type', 'pd.DataFrame'))
        p_desc = inp.get('description', 'Input DataFrame')
        docstring_lines.append(f'{p_name} ({p_type}): {p_desc}')
        docstring_lines.append(f'    role: input')

    for arg in args:
        p_name = arg['name']
        p_type = normalize_type(arg.get('type', 'any'))
        p_desc = arg.get('description', f'Parameter {p_name}')
        
        docstring_lines.append(f'{p_name} ({p_type}): {p_desc}')
        
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
        
        # Explicitly mark as parameter role if not specified
        role = arg.get('role', 'parameter')
        docstring_lines.append(f'    role: {role}')
        
    docstring_lines.append('')
    docstring_lines.append('Returns:')
    
    if not outputs:
        docstring_lines.append('None')
    else:
        for out in outputs:
            o_name = out.get("name", "output")
            o_type = normalize_type(out.get("type", "pd.DataFrame"))
            o_desc = out.get("description", "Result")
            docstring_lines.append(f'{o_name} ({o_type}): {o_desc}')
        
    docstring_str = "\n    ".join(docstring_lines)
    
    # Extract existing body if available
    body_str = f"    # Implementation\n    return {inputs[0]['name'] if inputs else 'pd.DataFrame()'}"
    if not outputs:
        body_str = "    # Implementation\n    pass"
    elif len(outputs) > 1:
         # Default return for multiple outputs
         ret_vals = []
         if inputs:
             # Just return input multiple times as placeholder?
             # Or pd.DataFrame()
             pass
         body_str = f"    # Implementation\n    return " + ", ".join(["pd.DataFrame()" for _ in outputs])

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

def {func_name}({sig_str}) {ret_annotation}:
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
        
        # Always extract imports from actual code (not from docstring)
        # This ensures single source of truth for imports
        imports = extract_imports_from_source(code)
        
        # Extract args from signature
        args = []
        inputs = []
        
        for arg in func_node.args.args:
            name = arg.arg
            
            # Skip self/cls if any (though these are functions)
            
            p_meta = params_meta.get(name, {})
            role = p_meta.get('role')
            
            # Get type from annotation if not in docstring
            arg_type = p_meta.get('type', 'any')
            if arg_type == 'any' and arg.annotation:
                 # Try to extract type from AST annotation
                 try:
                     if isinstance(arg.annotation, ast.Name):
                         arg_type = arg.annotation.id
                     elif isinstance(arg.annotation, ast.Attribute):
                         arg_type = arg.annotation.attr # e.g. DataFrame from pd.DataFrame
                 except:
                     pass

            # Infer role if not set
            if not role:
                if name == 'df' or 'df_' in name or 'dataframe' in str(arg_type).lower():
                    role = 'input'
                else:
                    role = 'parameter'
            
            item = {
                "name": name,
                "type": arg_type,
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
                
        # Outputs inference
        outputs = []
        
        if "Returns:" in docstring:
             parsed_returns = parse_docstring_returns(docstring)
             for i, ret in enumerate(parsed_returns):
                outputs.append({
                    "name": ret.get("name", f"output_{i}"),
                    "type": ret["type"],
                    "description": ret["description"]
                })
        else:
             # Default output for legacy code
             outputs.append({"name": "result", "type": "pd.DataFrame"})
            
        # Extract description (everything before the first section)
        desc_lines = []
        for line in docstring.split('\n'):
            stripped = line.strip()
            if stripped.startswith('Algorithm:') or stripped.startswith('Parameters:') or stripped.startswith('Returns:') or stripped.startswith('Example:'):
                break
            if stripped: # Only add non-empty lines, or preserve formatting?
                 # Let's preserve formatting but trim the line itself if needed?
                 # Usually docstrings are indented.
                 # Let's just strip for now as we rejoin with \n
                 desc_lines.append(stripped)
        
        description = "\n".join(desc_lines).strip()

        return {
             "id": func_node.name,
             "category": metadata.get('category'),
             "name": metadata.get('name'),
             "description": description,
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
