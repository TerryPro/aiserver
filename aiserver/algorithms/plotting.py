from .base import Algorithm, Port
from ..workflow_lib import plotting

# 箱型图绘制
box_plot = Algorithm(
    id="box_plot",
    name="箱型图绘制",
    category="数据绘图",
    prompt="请对 {VAR_NAME} 绘制箱型图，展示数据分布特征。支持单变量、多变量和分组箱型图。",
    algo_module=plotting,
    imports=["import pandas as pd", "import matplotlib.pyplot as plt", "import seaborn as sns"]
)

# 直方图显示
histogram = Algorithm(
    id="histogram",
    name="直方图显示",
    category="数据绘图",
    prompt="请对 {VAR_NAME} 绘制直方图，展示数据分布特征。支持自定义箱数、颜色、密度曲线等设置。",
    algo_module=plotting,
    imports=["import pandas as pd", "import matplotlib.pyplot as plt", "import seaborn as sns"]
)

# 密度图显示
density_plot = Algorithm(
    id="density_plot",
    name="密度图显示",
    category="数据绘图",
    prompt="请对 {VAR_NAME} 绘制密度图，展示数据分布特征。支持自定义颜色、带宽等设置。",
    algo_module=plotting,
    imports=["import pandas as pd", "import matplotlib.pyplot as plt", "import seaborn as sns"]
)

# 小提琴图显示
violin_plot = Algorithm(
    id="violin_plot",
    name="小提琴图显示",
    category="数据绘图",
    prompt="请对 {VAR_NAME} 绘制小提琴图，展示数据分布特征。支持自定义颜色、带宽等设置。",
    algo_module=plotting,
    imports=["import pandas as pd", "import matplotlib.pyplot as plt", "import seaborn as sns"]
)

# 相关性热力图
correlation_heatmap = Algorithm(
    id="correlation_heatmap",
    name="相关性热力图",
    category="数据绘图",
    prompt="请对 {VAR_NAME} 进行相关性分析。使用seaborn的heatmap函数，绘制各列之间的相关性热力图。",
    algo_module=plotting,
    imports=["import pandas as pd", "import matplotlib.pyplot as plt", "import seaborn as sns"]
)

algorithms = [box_plot, histogram, density_plot, violin_plot, correlation_heatmap]
