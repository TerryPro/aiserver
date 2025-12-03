from .base import Algorithm, Port
from ..workflow_lib import eda

autocorrelation = Algorithm(
    id="autocorrelation",
    name="自相关分析 (ACF)",
    category="eda",
    prompt="请对{VAR_NAME} 进行自相关分析。计算并绘制 ACF 图，使用 statsmodels.graphics.tsaplots.plot_acf，以发现周期性模式。",
    algo_module=eda,
    imports=["import matplotlib.pyplot as plt", "from statsmodels.graphics.tsaplots import plot_acf"],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")]
)

decomposition = Algorithm(
    id="decomposition",
    name="STL 分解",
    category="eda",
    prompt="请对{VAR_NAME} 执行 STL 分解 (Seasonal-Trend decomposition using LOESS)。将数据分解为趋势、季节与残差，并绘制分解结果图。",
    algo_module=eda,
    imports=["import matplotlib.pyplot as plt", "from statsmodels.tsa.seasonal import STL", "import pandas as pd"],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")]
)

sampling_period = Algorithm(
    id="sampling_period",
    name="采样周期统计",
    category="eda",
    prompt="请对{VAR_NAME} 进行采样周期统计。计算每一列数据的实际采样周次。",
    algo_module=eda,
    imports=["import pandas as pd"],
    inputs=[Port(name="df_in")],
    outputs=[],
)

data_features = Algorithm(
    id="data_features",
    name="数据特征",
    category="eda",
    prompt="请对{VAR_NAME} 进行数据特征计算。使用pandas的describe()函数，计算各列的基本统计特征。",
    algo_module=eda,
    imports=["import pandas as pd"],
    inputs=[Port(name="df_in")],
    outputs=[],
)

algorithms = [
    autocorrelation, decomposition, sampling_period, data_features
]
