from .base import Algorithm, Port
from ..workflow_lib import anomaly_detection

threshold_sigma = Algorithm(
    id="threshold_sigma",
    name="3-Sigma 异常检测",
    category="anomaly_detection",
    prompt="请对 {VAR_NAME} 执行 3-Sigma 异常检测。使用窗口大小 {window} 和阈值 {sigma}。",
    algo_module=anomaly_detection,
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    imports=["import pandas as pd", "import numpy as np"]
)

isolation_forest = Algorithm(
    id="isolation_forest",
    name="孤立森林异常检测",
    category="anomaly_detection",
    prompt="请对 {VAR_NAME} 执行孤立森林异常检测。设置异常比例为 {contamination}。",
    algo_module=anomaly_detection,
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    imports=["import pandas as pd", "from sklearn.ensemble import IsolationForest"]
)

change_point = Algorithm(
    id="change_point",
    name="变点检测",
    category="anomaly_detection",
    prompt="请对 {VAR_NAME} 执行变点检测。检测 {n_bkps} 个变点。",
    algo_module=anomaly_detection,
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    imports=["import pandas as pd", "import ruptures as rpt", "import matplotlib.pyplot as plt"]
)

zscore_anomaly = Algorithm(
    id="zscore_anomaly",
    name="Z-Score 异常检测",
    category="anomaly_detection",
    prompt="请对 {VAR_NAME} 执行 Z-Score 异常检测。使用阈值 {threshold}。",
    algo_module=anomaly_detection,
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    imports=["import pandas as pd", "import numpy as np"]
)

iqr_anomaly = Algorithm(
    id="iqr_anomaly",
    name="IQR 异常检测",
    category="anomaly_detection",
    prompt="请对 {VAR_NAME} 执行 IQR 异常检测。使用 IQR 倍数 {multiplier}。",
    algo_module=anomaly_detection,
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    imports=["import pandas as pd"]
)

moving_window_zscore_anomaly = Algorithm(
    id="moving_window_zscore_anomaly",
    name="移动窗口 Z-Score 检测",
    category="anomaly_detection",
    prompt="请对 {VAR_NAME} 执行移动窗口 Z-Score 异常检测。窗口大小 {window}，阈值 {threshold}。",
    algo_module=anomaly_detection,
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    imports=["import pandas as pd", "import numpy as np"]
)

moving_window_iqr_anomaly = Algorithm(
    id="moving_window_iqr_anomaly",
    name="移动窗口 IQR 检测",
    category="anomaly_detection",
    prompt="请对 {VAR_NAME} 执行移动窗口 IQR 异常检测。窗口大小 {window}，倍数 {multiplier}。",
    algo_module=anomaly_detection,
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    imports=["import pandas as pd"]
)

algorithms = [
    threshold_sigma, isolation_forest, change_point, zscore_anomaly, iqr_anomaly,
    moving_window_zscore_anomaly, moving_window_iqr_anomaly
]
