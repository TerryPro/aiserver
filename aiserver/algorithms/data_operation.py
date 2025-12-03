from .base import Algorithm, Port
from ..workflow_lib import data_operation

select_columns = Algorithm(
    id="select_columns",
    name="选择列",
    category="data_operation",
    prompt="请从 {VAR_NAME} 中选择指定的列 {columns}，生成新的 DataFrame。",
    algo_module=data_operation,
    imports=["import pandas as pd"]
)

filter_rows = Algorithm(
    id="filter_rows",
    name="筛选行",
    category="data_operation",
    prompt="请对 {VAR_NAME} 进行行筛选。根据条件 {condition} 筛选数据（例如 'age > 18'）。",
    algo_module=data_operation,
    imports=["import pandas as pd"]
)

concat_dfs = Algorithm(
    id="concat_dfs",
    name="数据连接 (Concat)",
    category="data_operation",
    prompt="请连接两个 DataFrame {df1} 和 {df2}。沿轴 {axis} 进行连接。",
    algo_module=data_operation,
    imports=["import pandas as pd"]
)

rename_columns = Algorithm(
    id="rename_columns",
    name="重命名列",
    category="data_operation",
    prompt="请对 {VAR_NAME} 的列进行重命名。使用映射关系 {columns_map}。",
    algo_module=data_operation,
    imports=["import pandas as pd"]
)

fill_na = Algorithm(
    id="fill_na",
    name="填充缺失值",
    category="data_operation",
    prompt="请对 {VAR_NAME} 填充缺失值。使用值 {value} 或方法 {method} 进行填充。",
    algo_module=data_operation,
    imports=["import pandas as pd", "import numpy as np"]
)

window_calculation = Algorithm(
    id="window_calculation",
    name="窗口计算",
    category="data_operation",
    prompt="请对 {VAR_NAME} 进行窗口计算。使用窗口大小 {window} 对列 {columns} 应用 {func} 函数。",
    algo_module=data_operation,
    imports=["import pandas as pd"]
)

merge_dfs = Algorithm(
    id="merge_dfs",
    name="数据合并 (Merge)",
    category="data_operation",
    prompt="请合并两个数据框 {left} 和 {right}。根据指定的合并方式（inner, outer, left, right）和连接键进行 pd.merge 操作。",
    algo_module=data_operation,
    imports=["import pandas as pd"]
)

algorithms = [
    select_columns, filter_rows, concat_dfs, rename_columns, fill_na, window_calculation, merge_dfs
]
