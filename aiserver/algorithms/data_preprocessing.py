from .base import Algorithm, AlgorithmParameter, Port
from ..workflow_lib import data_preprocessing

interpolation_spline = Algorithm(
    id="interpolation_spline",
    name="样条插值",
    category="data_preprocessing",
    prompt="请对{VAR_NAME} 进行样条插值 (Spline)。使用 pandas 的 interpolate(method='spline', order=3) 以获得更平滑的补全曲线。",
    algo_module=data_preprocessing,
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    imports=["import pandas as pd", "import numpy as np"]
)

resampling_down = Algorithm(
    id="resampling_down",
    name="降采样",
    category="data_preprocessing",
    prompt="请对{VAR_NAME} 进行降采样聚合。使用 pandas 的 resample() 将数据聚合到更低的时间分辨率（例如 '1min' 或 '1H'）；数值列使用 mean()，状态列使用 last() 或 max()。",
    algo_module=data_preprocessing,
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    imports=["import pandas as pd"]
)

alignment = Algorithm(
    id="alignment",
    name="多源数据对齐",
    category="data_preprocessing",
    prompt="请以 {VAR_NAME} 为基准执行多源时间对齐。使用 pandas 的 merge_asof 方法，将其他数据对齐到该时间轴。",
    algo_module=data_preprocessing,
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    imports=["import pandas as pd"]
)

feature_scaling = Algorithm(
    id="feature_scaling",
    name="数据标准化/归一化",
    category="data_preprocessing",
    prompt="请对 {VAR_NAME} 进行特征缩放。支持多种缩放方法，直接修改原始列。",
    algo_module=data_preprocessing,
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    imports=["import pandas as pd", "from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler"]
)

diff_transform = Algorithm(
    id="diff_transform",
    name="差分变换",
    category="data_preprocessing",
    prompt="请对 {VAR_NAME} 进行差分变换，以消除趋势并使数据平稳。可配置差分阶数和滞后步数。",
    algo_module=data_preprocessing,
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    imports=["import pandas as pd"]
)

data_fill = Algorithm(
    id="data_fill",
    name="数据填充",
    category="data_preprocessing",
    prompt="请对 {VAR_NAME} 进行缺失值填充。支持多种填充方法，包括均值、中位数、众数、前向填充、后向填充、常数填充等。",
    algo_module=data_preprocessing,
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    imports=["import pandas as pd"]
)

algorithms = [
    interpolation_spline, resampling_down, alignment, feature_scaling, diff_transform, data_fill
]
