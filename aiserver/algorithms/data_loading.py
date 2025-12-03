from .base import Algorithm, AlgorithmParameter, Port
from ..workflow_lib import data_loading

load_csv = Algorithm(
    id="load_csv",
    name="加载 CSV",
    category="load_data",
    prompt="请加载 CSV 文件。使用 pandas.read_csv 读取指定路径的文件，并显示前几行数据。",
    algo_module=data_loading,
    imports=["import pandas as pd", "import os"],
    inputs=[],  # 加载CSV节点没有输入（起始节点）
    outputs=[Port(name="df_out")]  # 加载CSV节点有输出
    # template 由动态生成，无需硬编码
)

import_variable = Algorithm(
    id="import_variable",
    name="引入变量",
    category="load_data",
    prompt="请引入已存在的 DataFrame 变量 {variable_name}。创建其副本或引用，以便后续分析使用。",
    algo_module=data_loading,
    imports=["import pandas as pd"],
    inputs=[],  # 引入变量节点没有输入
    outputs=[Port(name="df_out")]  # 引入变量节点有输出
    # template 由动态生成，无需硬编码
)

export_data = Algorithm(
    id="export_data",
    name="引出变量",
    category="load_data",
    prompt="请将 {VAR_NAME} 引出到全局环境。变量名为 {global_name}。",
    algo_module=data_loading,
    imports=[],
    inputs=[Port(name="df_in")],  # 引出变量节点有输入但无输出
    outputs=[],  # 引出变量节点无输出
    # template 由动态生成，无需硬编码
)

algorithms = [load_csv, import_variable, export_data]
