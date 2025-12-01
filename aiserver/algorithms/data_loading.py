from .base import Algorithm, AlgorithmParameter

load_csv = Algorithm(
    id="load_csv",
    name="加载 CSV",
    category="load_data",
    prompt="请加载 CSV 文件。使用 pandas.read_csv 读取指定路径的文件，并显示前几行数据。",
    parameters=[
        AlgorithmParameter(
            name="filepath",
            type="str",
            default="dataset/data.csv",
            label="文件路径",
            description="CSV文件路径 (相对于项目根目录)",
            widget="file-selector"
        )
    ],
    imports=["import pandas as pd", "import os"],
    template="""
# Load CSV Data
filepath = '{filepath}'
if not os.path.exists(filepath):
    print(f"Error: File not found at {filepath}")
else:
    {OUTPUT_VAR} = pd.read_csv(filepath)
    print(f"Loaded {OUTPUT_VAR} with shape: {{OUTPUT_VAR}.shape}")
    display({OUTPUT_VAR}.head())
"""
)

import_variable = Algorithm(
    id="import_variable",
    name="引入变量",
    category="load_data",
    prompt="请引入已存在的 DataFrame 变量 {variable_name}。创建其副本或引用，以便后续分析使用。",
    parameters=[
        AlgorithmParameter(
            name="variable_name",
            type="str",
            default="",
            label="变量名称",
            description="当前会话中的DataFrame变量名",
            widget="variable-selector"
        )
    ],
    imports=["import pandas as pd"],
    template="""
# Import Existing Variable
# Source: {variable_name}
# Output: {OUTPUT_VAR}

{OUTPUT_VAR} = None
try:
    # Check in globals() because locals() inside a method won't find notebook global variables
    if '{variable_name}' not in globals():
        print(f"Error: Variable '{variable_name}' not found in global scope.")
    else:
        # Create a copy to avoid modifying the original variable accidentally
        # We must access the variable via globals()['name'] or directly if we could eval it, 
        # but since we are inside a function, simple reference might fail if not passed in.
        # However, in Python, reading a global variable inside a function works if not shadowed.
        # But 'if name in globals()' is the check.
        
        source_var = globals()['{variable_name}']
        if hasattr(source_var, 'copy'):
            {OUTPUT_VAR} = source_var.copy()
        else:
            {OUTPUT_VAR} = source_var # Shallow copy/ref if copy() not available
            
        print(f"Imported '{variable_name}' as '{OUTPUT_VAR}'")
        if hasattr({OUTPUT_VAR}, 'shape'):
             print(f"Shape: {{{OUTPUT_VAR}.shape}}")
        if hasattr({OUTPUT_VAR}, 'head'):
             display({OUTPUT_VAR}.head())
except Exception as e:
    print(f"Import failed: {e}")
"""
)

export_data = Algorithm(
    id="export_data",
    name="引出变量",
    category="load_data",
    prompt="请将 {VAR_NAME} 引出到全局环境。变量名为 {global_name}。",
    parameters=[
        AlgorithmParameter(
            name="global_name",
            type="str",
            default="exported_data",
            label="全局变量名",
            description="引出的全局变量名称"
        )
    ],
    imports=[],
    template="""
# Export {VAR_NAME} to Global Variable
# Variable Name: {global_name}
# This node acts as a pass-through in the flow
{OUTPUT_VAR} = {VAR_NAME}
print(f"Marked {VAR_NAME} for export to global variable: '{global_name}'")
"""
)

algorithms = [load_csv, import_variable, export_data]
