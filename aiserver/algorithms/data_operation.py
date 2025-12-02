from .base import Algorithm, AlgorithmParameter, Port

select_columns = Algorithm(
    id="select_columns",
    name="选择列",
    category="data_operation",
    prompt="请从 {VAR_NAME} 中选择指定的列 {columns}，生成新的 DataFrame。",
    parameters=[
        AlgorithmParameter(name="columns", type="list", default=[], label="选择列", description="要选择的列列表", widget="column-selector")
    ],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    imports=["import pandas as pd"],
    template="""
# Select Columns for {VAR_NAME}
{OUTPUT_VAR} = {VAR_NAME}.copy()
columns_to_select = {columns}

# Check if columns exist
missing_cols = [c for c in columns_to_select if c not in {OUTPUT_VAR}.columns]
if missing_cols:
    print(f"Error: Columns not found: {missing_cols}")
else:
    {OUTPUT_VAR} = {OUTPUT_VAR}[columns_to_select]
    print(f"Selected {len(columns_to_select)} columns.")
    display({OUTPUT_VAR}.head())
"""
)

filter_rows = Algorithm(
    id="filter_rows",
    name="筛选行",
    category="data_operation",
    prompt="请对 {VAR_NAME} 进行行筛选。根据条件 {condition} 筛选数据（例如 'age > 18'）。",
    parameters=[
        AlgorithmParameter(name="condition", type="str", default="", label="过滤条件", description="查询字符串 (例如 'age > 18')")
    ],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    imports=["import pandas as pd"],
    template="""
# Filter Rows for {VAR_NAME}
{OUTPUT_VAR} = {VAR_NAME}.copy()
condition = "{condition}"

try:
    {OUTPUT_VAR} = {OUTPUT_VAR}.query(condition)
    print(f"Filtered rows using query: '{condition}'")
    print(f"Rows remaining: {{OUTPUT_VAR}.shape[0]}")
    display({OUTPUT_VAR}.head())
except Exception as e:
    print(f"Filtering failed: {e}")
"""
)

concat_dfs = Algorithm(
    id="concat_dfs",
    name="数据连接 (Concat)",
    category="data_operation",
    prompt="请连接两个 DataFrame {df1} 和 {df2}。沿轴 {axis} 进行连接。",
    parameters=[
        AlgorithmParameter(name="axis", type="int", default=0, label="轴向", description="拼接轴向 (0=行, 1=列)")
    ],
    inputs=[Port(name="df1"), Port(name="df2")],
    outputs=[Port(name="df_out")],
    imports=["import pandas as pd"],
    template="""
# Concat DataFrames
# Inputs: {df1}, {df2}
# Output: {OUTPUT_VAR}

try:
    axis = {axis}
    {OUTPUT_VAR} = pd.concat([{df1}, {df2}], axis=axis)
    print(f"Concatenated DataFrames along axis {axis}.")
    print(f"New shape: {{OUTPUT_VAR}.shape}")
    display({OUTPUT_VAR}.head())
except Exception as e:
    print(f"Concat failed: {e}")
"""
)

rename_columns = Algorithm(
    id="rename_columns",
    name="重命名列",
    category="data_operation",
    prompt="请对 {VAR_NAME} 的列进行重命名。使用映射关系 {columns_map}。",
    parameters=[
        AlgorithmParameter(name="columns_map", type="dict", default={}, label="列名映射", description="旧名到新名的映射字典")
    ],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    imports=["import pandas as pd"],
    template="""
# Rename Columns for {VAR_NAME}
{OUTPUT_VAR} = {VAR_NAME}.copy()
columns_map = {columns_map}

try:
    {OUTPUT_VAR} = {OUTPUT_VAR}.rename(columns=columns_map)
    print(f"Renamed columns using map: {columns_map}")
    display({OUTPUT_VAR}.head())
except Exception as e:
    print(f"Renaming failed: {e}")
"""
)

fill_na = Algorithm(
    id="fill_na",
    name="填充缺失值",
    category="data_operation",
    prompt="请对 {VAR_NAME} 填充缺失值。使用值 {value} 或方法 {method} 进行填充。",
    parameters=[
        AlgorithmParameter(name="value", type="str", default=None, label="填充值", description="用于填充的常数值 (可选)"),
        AlgorithmParameter(name="method", type="enum", default=None, label="填充方法", options=["ffill", "bfill"], description="填充方法 (可选)")
    ],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    imports=["import pandas as pd", "import numpy as np"],
    template="""
# Fill Missing Values for {VAR_NAME}
{OUTPUT_VAR} = {VAR_NAME}.copy()
value = {value}
method = '{method}'

try:
    if value is not None and value != '':
        {OUTPUT_VAR} = {OUTPUT_VAR}.fillna(value=value)
        print(f"Filled NA with value: {value}")
    elif method:
        {OUTPUT_VAR} = {OUTPUT_VAR}.fillna(method=method)
        print(f"Filled NA with method: {method}")
    else:
        print("Warning: No value or method specified for fillna.")
        
    display({OUTPUT_VAR}.head())
except Exception as e:
    print(f"Fill NA failed: {e}")
"""
)

window_calculation = Algorithm(
    id="window_calculation",
    name="窗口计算",
    category="data_operation",
    prompt="请对 {VAR_NAME} 进行窗口计算。使用窗口大小 {window} 对列 {columns} 应用 {func} 函数。",
    parameters=[
        AlgorithmParameter(name="columns", type="list", default=[], label="计算列", description="要计算的列，为空则使用所有数值列", widget="column-selector", priority="critical"),
        AlgorithmParameter(name="window", type="int", default=10, label="窗口大小", description="移动窗口的大小", priority="critical"),
        AlgorithmParameter(name="func", type="str", default="mean", label="统计函数", options=["sum", "mean", "min", "max", "std", "var"], description="要应用的统计函数", priority="critical"),
        AlgorithmParameter(name="min_periods", type="int", default=1, label="最小观测值", description="窗口中需要的最小观测值数量", priority="non-critical"),
        AlgorithmParameter(name="center", type="bool", default=False, label="居中窗口", description="是否居中窗口", priority="non-critical")
    ],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    imports=["import pandas as pd"],
    template="""
# Window Calculation for {VAR_NAME}
{OUTPUT_VAR} = {VAR_NAME}.copy()
columns = {columns}
window_size = {window}
func = '{func}'
min_periods = {min_periods}
center = {center}

# Select columns if specified, otherwise use all numeric columns
if not columns:
    columns = {OUTPUT_VAR}.select_dtypes(include=['number']).columns.tolist()

# Apply rolling window function
try:
    for col in columns:
        {OUTPUT_VAR}[col] = {OUTPUT_VAR}[col].rolling(
            window=window_size,
            min_periods=min_periods,
            center=center
        ).{func}()
    
    print(f"Applied {func} with window size {window_size} to columns: {columns}")
    display({OUTPUT_VAR}.head())
except Exception as e:
    print(f"Window calculation failed: {e}")
"""
)

merge_dfs = Algorithm(
    id="merge_dfs",
    name="数据合并 (Merge)",
    category="data_operation",
    prompt="请合并两个数据框 {left} 和 {right}。根据指定的合并方式（inner, outer, left, right）和连接键进行 pd.merge 操作。",
    parameters=[
        AlgorithmParameter(name="how", type="str", default="inner", label="合并方式", options=["inner", "outer", "left", "right"], description="执行合并的方式"),
        AlgorithmParameter(name="on", type="str", default="", label="合并列", description="用于连接的列名或索引级别名。留空则使用索引。", widget="column-selector")
    ],
    inputs=[Port(name="left"), Port(name="right")],
    outputs=[Port(name="merged")],
    imports=["import pandas as pd"],
    template="""
# Merge DataFrames
# Inputs: {left}, {right}
# Output: {merged}



try:
    # Check if inputs are available
    if '{left}' not in locals() or '{right}' not in locals():
        print("Error: Input DataFrames not found.")
    else:
        on_col = '{on}'
        if on_col == '':
            # Merge on index if no column specified
            {merged} = pd.merge({left}, {right}, how='{how}', left_index=True, right_index=True)
        else:
            {merged} = pd.merge({left}, {right}, how='{how}', on=on_col)
            
        print(f"Merged shape: {{merged}.shape}")
        display({merged}.head())
except Exception as e:
    print(f"Merge failed: {e}")
"""
)

algorithms = [
    select_columns, filter_rows, concat_dfs, rename_columns, fill_na, window_calculation, merge_dfs
]