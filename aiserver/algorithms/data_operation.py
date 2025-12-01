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

sort_values = Algorithm(
    id="sort_values",
    name="排序",
    category="data_operation",
    prompt="请对 {VAR_NAME} 进行排序。根据列 {by} 进行 {ascending} 排序。",
    parameters=[
        AlgorithmParameter(name="by", type="list", default=[], label="排序依据", description="排序的依据列", widget="column-selector"),
        AlgorithmParameter(name="ascending", type="bool", default=True, label="升序", description="升序还是降序")
    ],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    imports=["import pandas as pd"],
    template="""
# Sort Values for {VAR_NAME}
{OUTPUT_VAR} = {VAR_NAME}.copy()
by_columns = {by}
ascending = {ascending}

try:
    {OUTPUT_VAR} = {OUTPUT_VAR}.sort_values(by=by_columns, ascending=ascending)
    print(f"Sorted by {by_columns} (Ascending: {ascending})")
    display({OUTPUT_VAR}.head())
except Exception as e:
    print(f"Sorting failed: {e}")
"""
)

groupby_agg = Algorithm(
    id="groupby_agg",
    name="分组聚合",
    category="data_operation",
    prompt="请对 {VAR_NAME} 进行分组聚合。按 {by} 分组，并对 {agg_dict} 执行聚合操作。",
    parameters=[
        AlgorithmParameter(name="by", type="list", default=[], label="分组依据", description="分组的依据列", widget="column-selector"),
        AlgorithmParameter(name="agg_dict", type="dict", default={}, label="聚合字典", description="聚合配置字典 (例如 {'col': 'mean'})")
    ],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    imports=["import pandas as pd", "import numpy as np"],
    template="""
# GroupBy Aggregation for {VAR_NAME}
{OUTPUT_VAR} = {VAR_NAME}.copy()
by_columns = {by}
agg_dict = {agg_dict}

try:
    {OUTPUT_VAR} = {OUTPUT_VAR}.groupby(by_columns).agg(agg_dict)
    print(f"Grouped by {by_columns} and aggregated.")
    display({OUTPUT_VAR}.head())
except Exception as e:
    print(f"GroupBy failed: {e}")
"""
)

pivot_table = Algorithm(
    id="pivot_table",
    name="透视表",
    category="data_operation",
    prompt="请对 {VAR_NAME} 创建透视表。索引={index}, 列={columns}, 值={values}, 聚合函数={aggfunc}。",
    parameters=[
        AlgorithmParameter(name="values", type="str", default="", label="值列", description="要聚合的列", widget="column-selector"),
        AlgorithmParameter(name="index", type="list", default=[], label="索引列", description="索引列列表", widget="column-selector"),
        AlgorithmParameter(name="columns", type="list", default=[], label="列名列", description="列名列列表", widget="column-selector"),
        AlgorithmParameter(name="aggfunc", type="str", default="mean", label="聚合函数", description="聚合函数")
    ],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    imports=["import pandas as pd", "import numpy as np"],
    template="""
# Pivot Table for {VAR_NAME}
{OUTPUT_VAR} = {VAR_NAME}.copy()

try:
    {OUTPUT_VAR} = pd.pivot_table(
        {OUTPUT_VAR}, 
        values={values}, 
        index={index}, 
        columns={columns}, 
        aggfunc={aggfunc}
    )
    print("Pivot table created.")
    display({OUTPUT_VAR}.head())
except Exception as e:
    print(f"Pivot table failed: {e}")
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

drop_duplicates = Algorithm(
    id="drop_duplicates",
    name="去重",
    category="data_operation",
    prompt="请对 {VAR_NAME} 去除重复行。基于列 {subset} 判断重复，保留 {keep}。",
    parameters=[
        AlgorithmParameter(name="subset", type="list", default=[], label="子集", description="考虑重复的列子集", widget="column-selector"),
        AlgorithmParameter(name="keep", type="str", default="first", label="保留策略", options=["first", "last", "False"], description="保留哪个重复项")
    ],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    imports=["import pandas as pd"],
    template="""
# Drop Duplicates for {VAR_NAME}
{OUTPUT_VAR} = {VAR_NAME}.copy()
subset = {subset}
keep = '{keep}'

try:
    initial_rows = {OUTPUT_VAR}.shape[0]
    if not subset:
        subset = None
    {OUTPUT_VAR} = {OUTPUT_VAR}.drop_duplicates(subset=subset, keep=keep)
    dropped_rows = initial_rows - {OUTPUT_VAR}.shape[0]
    print(f"Dropped {dropped_rows} duplicate rows.")
    display({OUTPUT_VAR}.head())
except Exception as e:
    print(f"Drop duplicates failed: {e}")
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

astype = Algorithm(
    id="astype",
    name="类型转换",
    category="data_operation",
    prompt="请对 {VAR_NAME} 的列进行类型转换。转换规则为 {dtype_map}。",
    parameters=[
        AlgorithmParameter(name="dtype_map", type="dict", default={}, label="类型映射", description="列到类型的映射字典")
    ],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    imports=["import pandas as pd"],
    template="""
# Change Column Types for {VAR_NAME}
{OUTPUT_VAR} = {VAR_NAME}.copy()
dtype_map = {dtype_map}

try:
    {OUTPUT_VAR} = {OUTPUT_VAR}.astype(dtype_map)
    print(f"Converted column types using map: {dtype_map}")
    display({OUTPUT_VAR}.dtypes)
except Exception as e:
    print(f"Type conversion failed: {e}")
"""
)

apply_func = Algorithm(
    id="apply_func",
    name="自定义函数 (Apply)",
    category="data_operation",
    prompt="请对 {VAR_NAME} 应用自定义函数。对 {axis} 轴应用函数 {func_code}。",
    parameters=[
        AlgorithmParameter(name="func_code", type="str", default="lambda x: x", label="函数代码", description="函数或lambda的Python代码"),
        AlgorithmParameter(name="axis", type="int", default=0, label="轴向", description="应用轴向 (0或1)")
    ],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    imports=["import pandas as pd", "import numpy as np"],
    template="""
# Apply Function for {VAR_NAME}
{OUTPUT_VAR} = {VAR_NAME}.copy()
axis = {axis}
# Define the function here or use lambda in func_code
# Example func_code: "lambda x: x.sum()" or "np.sum"

try:
    func = {func_code}
    {OUTPUT_VAR} = {OUTPUT_VAR}.apply(func, axis=axis)
    print(f"Applied function along axis {axis}.")
    # Note: Output might be Series or DataFrame depending on result
    if isinstance({OUTPUT_VAR}, pd.Series):
        {OUTPUT_VAR} = {OUTPUT_VAR}.to_frame()
    display({OUTPUT_VAR}.head())
except Exception as e:
    print(f"Apply failed: {e}")
"""
)

algorithms = [
    select_columns, filter_rows, sort_values, groupby_agg, pivot_table,
    concat_dfs, rename_columns, drop_duplicates, fill_na, astype, apply_func
]









