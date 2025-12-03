from .base import Algorithm, AlgorithmParameter, Port
from ..workflow_lib import trend
# from .utils import extract_parameters_from_func # 不再需要

trend_plot = Algorithm(
    id="trend_plot",
    name="通用趋势图 (Trend)",
    category="trend_plot",
    prompt="请根据配置绘制 {VAR_NAME} 的趋势图。支持自定义 X 轴、Y 轴列、标题、网格等设置。",
    algo_module=trend,
    imports=["import matplotlib.pyplot as plt", "import pandas as pd", "import math"],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")]
)

trend_ma = Algorithm(
    id="trend_ma",
    name="移动平均趋势",
    category="trend_plot",
    prompt="请对{VAR_NAME} 绘制移动平均趋势线。使用 pandas 的 rolling().mean() 计算趋势线，并用 matplotlib 绘制原始曲线与趋势线，添加网格、图例与中文标签。",
    algo_module=trend,
    imports=["import pandas as pd", "import matplotlib.pyplot as plt"],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")]
)

trend_ewma = Algorithm(
    id="trend_ewma",
    name="指数加权趋势",
    category="trend_plot",
    prompt="请对{VAR_NAME} 绘制 EWMA（指数加权移动平均）趋势线。使用 pandas 的 ewm(span=...).mean() 计算趋势，并使用 matplotlib 将原始数据与 EWMA 趋势曲线叠加展示。",
    algo_module=trend,
    imports=["import pandas as pd", "import matplotlib.pyplot as plt"],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")]
)

trend_loess = Algorithm(
    id="trend_loess",
    name="LOESS 趋势",
    category="trend_plot",
    prompt="请对{VAR_NAME} 绘制 LOESS 平滑趋势。使用 statsmodels.nonparametric.smoothers_lowess.lowess 进行平滑并绘制趋势曲线；若缺少该库，可退化为 rolling().mean()。",
    algo_module=trend,
    imports=["import statsmodels.api as sm", "import matplotlib.pyplot as plt", "import numpy as np", "import pandas as pd"],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")]
)

trend_polyfit = Algorithm(
    id="trend_polyfit",
    name="多项式趋势拟合",
    category="trend_plot",
    prompt="请对{VAR_NAME} 进行多项式趋势拟合并绘制趋势。使用 numpy.polyfit 对指定阶数进行拟合，绘制拟合曲线与原始数据，并计算与输出拟合优度（R²）。",
    algo_module=trend,
    imports=["import numpy as np", "import matplotlib.pyplot as plt", "import pandas as pd"],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")]
)

trend_stl_trend = Algorithm(
    id="trend_stl_trend",
    name="STL 趋势分量",
    category="trend_plot",
    prompt="请对{VAR_NAME} 执行 STL 分解并提取趋势分量。使用 statsmodels.tsa.seasonal.STL 提取趋势，绘制趋势曲线并与原始数据对比显示。",
    algo_module=trend,
    imports=["from statsmodels.tsa.seasonal import STL", "import matplotlib.pyplot as plt", "import pandas as pd"],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")]
)

trend_ohlc = Algorithm(
    id="trend_ohlc",
    name="OHLC重采样",
    category="trend_plot",
    prompt="请对{VAR_NAME} 进行OHLC重采样。将时间序列数据重采样为指定频率的开盘价(Open)、最高价(High)、最低价(Low)和收盘价(Close)，并绘制蜡烛图。",
    algo_module=trend,
    imports=["import pandas as pd", "import matplotlib.pyplot as plt", "import mplfinance.original_flavor as mpf"],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")]
)

trend_envelope = Algorithm(
    id="trend_envelope",
    name="数据包络线绘制",
    category="trend_plot",
    prompt="请对{VAR_NAME} 绘制数据包络线。使用滚动窗口的最大值和最小值计算上、下包络线，并与原始曲线一起绘制。",
    algo_module=trend,
    imports=["import pandas as pd", "import matplotlib.pyplot as plt", "import numpy as np"],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")]
)

algorithms = [
    trend_plot, trend_ma, trend_ewma, trend_loess, trend_polyfit,
    trend_stl_trend, trend_ohlc, trend_envelope
]
