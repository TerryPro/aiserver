from .base import Algorithm, AlgorithmParameter

trend_plot = Algorithm(
    id="trend_plot",
    name="通用趋势图 (Trend)",
    category="trend_plot",
    prompt="请根据配置绘制 {VAR_NAME} 的趋势图。支持自定义 X 轴、Y 轴列、标题、网格等设置。",
    parameters=[
        AlgorithmParameter(name="x_column", type="str", default="", label="X轴列名", description="作为X轴的列 (留空则使用索引)", widget="column-selector"),
        AlgorithmParameter(name="y_columns", type="list", default=[], label="Y轴列名", description="Y轴数据列 (留空则绘制所有数值列)", widget="column-selector"),
        AlgorithmParameter(name="title", type="str", default="趋势图", label="图表标题", description="图表的标题"),
        AlgorithmParameter(name="xlabel", type="str", default="", label="X轴标签", description="X轴的显示标签"),
        AlgorithmParameter(name="ylabel", type="str", default="", label="Y轴标签", description="Y轴的显示标签"),
        AlgorithmParameter(name="grid", type="bool", default=True, label="显示网格", description="是否显示背景网格"),
        AlgorithmParameter(name="figsize", type="str", default="(10, 6)", label="图像尺寸", description="图像大小元组，例如 (10, 6)")
    ],
    imports=["import matplotlib.pyplot as plt", "import pandas as pd"],
    template="""
# Trend Plot (Complex)
# Input: {VAR_NAME}
# Output: {OUTPUT_VAR} (Pass-through)

import matplotlib.pyplot as plt
import pandas as pd

{OUTPUT_VAR} = {VAR_NAME}

try:
    # Configuration
    x_col = '{x_column}'
    y_cols_str = '{y_columns}'
    title = '{title}'
    xlabel = '{xlabel}'
    ylabel = '{ylabel}'
    show_grid = {grid}
    figsize_str = '{figsize}'
    
    # Parse figsize
    try:
        figsize = eval(figsize_str) if figsize_str else (10, 6)
    except:
        figsize = (10, 6)

    plt.figure(figsize=figsize)
    
    # Plotting
    if x_col and x_col in {VAR_NAME}.columns:
        # Convert to datetime if possible for better plotting
        try:
            x_data = pd.to_datetime({VAR_NAME}[x_col])
        except Exception:
            print(f"Warning: Could not convert column '{x_col}' to datetime. Using original values.")
            x_data = {VAR_NAME}[x_col]
    else:
        x_data = {VAR_NAME}.index
        if x_col:
            print(f"Warning: X column '{x_col}' not found, using index.")

    # Parse Y columns
    y_cols = []
    if y_cols_str:
        # Handle list string representation like "['a', 'b']" or "[]"
        s = y_cols_str.strip()
        if s.startswith('[') and s.endswith(']'):
            s = s[1:-1]
        y_cols = [c.strip().strip("'").strip('"') for c in s.split(',') if c.strip()]

    if not y_cols:
        # If empty (or parsed as empty), fallback to numeric columns only
        y_cols = {VAR_NAME}.select_dtypes(include=['number']).columns.tolist()
    
    # Remove x_col from y_cols if present to avoid plotting time/index on Y-axis
    if x_col and x_col in y_cols:
        y_cols.remove(x_col)

    for col in y_cols:
        if col in {VAR_NAME}.columns:
            plt.plot(x_data, {VAR_NAME}[col], label=col)
        else:
            print(f"Warning: Y column '{col}' not found.")

    # Support Chinese characters in title and labels if needed
    plt.rcParams['font.sans-serif'] = ['SimHei'] # Use SimHei for Chinese
    plt.rcParams['axes.unicode_minus'] = False   # Fix minus sign

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(show_grid)
    plt.legend()
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"Error creating trend plot: {e}")
"""
)

trend_ma = Algorithm(
    id="trend_ma",
    name="移动平均趋势",
    category="trend_plot",
    prompt="请对{VAR_NAME} 绘制移动平均趋势线。先推断采样频率并将数据重采样到统一时间轴（如 '1S'），选择合理的窗口长度（例如 60 或 300 秒），使用 pandas 的 rolling().mean() 计算趋势线，并用 matplotlib 绘制原始曲线与趋势线，添加网格、图例与中文标签。若 {VAR_NAME} 为 DataFrame，请对数值列分别绘制。",
    parameters=[],
    imports=["import matplotlib.pyplot as plt"],
    template="""
# Moving Average Trend for {VAR_NAME}
df_trend = {VAR_NAME}.select_dtypes(include=['number']).copy()
target_col = df_trend.columns[0]

window = 50
df_trend['Trend'] = df_trend[target_col].rolling(window=window, center=True).mean()

plt.figure(figsize=(14, 6))
plt.plot(df_trend.index, df_trend[target_col], label='Original', alpha=0.4)
plt.plot(df_trend.index, df_trend['Trend'], label=f'MA (window={window})', linewidth=2, color='red')
plt.title(f'Moving Average Trend: {target_col}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
"""
)

trend_ewma = Algorithm(
    id="trend_ewma",
    name="指数加权趋势",
    category="trend_plot",
    prompt="请对{VAR_NAME} 绘制 EWMA（指数加权移动平均）趋势线。统一时间轴后，依据采样频率选择合适的 span（如 60 或 300），使用 pandas 的 ewm(span=...).mean() 计算趋势，并使用 matplotlib 将原始数据与 EWMA 趋势曲线叠加展示。",
    parameters=[],
    imports=["import matplotlib.pyplot as plt"],
    template="""
# EWMA Trend for {VAR_NAME}
df_trend = {VAR_NAME}.select_dtypes(include=['number']).copy()
target_col = df_trend.columns[0]

span = 50
df_trend['EWMA'] = df_trend[target_col].ewm(span=span).mean()

plt.figure(figsize=(14, 6))
plt.plot(df_trend.index, df_trend[target_col], label='Original', alpha=0.4)
plt.plot(df_trend.index, df_trend['EWMA'], label=f'EWMA (span={span})', linewidth=2, color='orange')
plt.title(f'Exponential Weighted Moving Average: {target_col}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
"""
)

trend_loess = Algorithm(
    id="trend_loess",
    name="LOESS 趋势",
    category="trend_plot",
    prompt="请对{VAR_NAME} 绘制 LOESS 平滑趋势。将时间序列统一到同一采样频率后，使用 statsmodels.nonparametric.smoothers_lowess.lowess 进行平滑并绘制趋势曲线；若缺少该库，可退化为 rolling().mean()。图表需包含中文标题、轴标签与图例。",
    parameters=[],
    imports=["import statsmodels.api as sm", "import matplotlib.pyplot as plt", "import numpy as np"],
    template="""
# LOESS Trend for {VAR_NAME}
df_trend = {VAR_NAME}.select_dtypes(include=['number']).dropna().copy()
target_col = df_trend.columns[0]

# Lowess requires numeric x-axis
x = np.arange(len(df_trend))
y = df_trend[target_col].values

# frac controls smoothing amount (between 0 and 1)
lowess = sm.nonparametric.lowess(y, x, frac=0.1)

plt.figure(figsize=(14, 6))
plt.plot(df_trend.index, y, label='Original', alpha=0.4)
plt.plot(df_trend.index, lowess[:, 1], label='LOESS', linewidth=2, color='green')
plt.title(f'LOESS Trend: {target_col}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
"""
)

trend_polyfit = Algorithm(
    id="trend_polyfit",
    name="多项式趋势拟合",
    category="trend_plot",
    prompt="请对{VAR_NAME} 进行多项式趋势拟合并绘制趋势。将时间戳转换为连续时间序列（秒或索引），使用 numpy.polyfit 对 1~2 阶进行拟合，绘制拟合曲线与原始数据，并计算与输出拟合优度（R²）。",
    parameters=[],
    imports=["import numpy as np", "import matplotlib.pyplot as plt"],
    template="""
# Polynomial Trend Fit for {VAR_NAME}
df_trend = {VAR_NAME}.select_dtypes(include=['number']).dropna().copy()
target_col = df_trend.columns[0]

y = df_trend[target_col].values
x = np.arange(len(y))

# Fit 2nd degree polynomial
coefs = np.polyfit(x, y, deg=2)
trend_poly = np.polyval(coefs, x)

plt.figure(figsize=(14, 6))
plt.plot(df_trend.index, y, label='Original', alpha=0.4)
plt.plot(df_trend.index, trend_poly, label=f'Poly Fit (deg=2)', linewidth=2, color='purple')
plt.title(f'Polynomial Trend Fit: {target_col}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
"""
)

trend_stl_trend = Algorithm(
    id="trend_stl_trend",
    name="STL 趋势分量",
    category="trend_plot",
    prompt="请对{VAR_NAME} 执行 STL 分解并提取趋势分量。统一采样频率后，使用 statsmodels.tsa.seasonal.STL 提取趋势，绘制趋势曲线并与原始数据对比显示；根据卫星遥测的特性选择合适的季节周期（如日照周期）。",
    parameters=[],
    imports=["from statsmodels.tsa.seasonal import STL", "import matplotlib.pyplot as plt", "import pandas as pd"],
    template="""
# STL Trend Extraction for {VAR_NAME}
df_stl = {VAR_NAME}.select_dtypes(include=['number']).copy()

# Handle frequency
if isinstance(df_stl.index, pd.DatetimeIndex) and df_stl.index.freq is None:
    inferred_freq = pd.infer_freq(df_stl.index)
    if inferred_freq:
        df_stl = df_stl.asfreq(inferred_freq).interpolate()

target_col = df_stl.columns[0]

try:
    res = STL(df_stl[target_col], robust=True).fit()
    
    plt.figure(figsize=(14, 6))
    plt.plot(df_stl.index, df_stl[target_col], label='Original', alpha=0.4)
    plt.plot(df_stl.index, res.trend, label='STL Trend', linewidth=2, color='brown')
    plt.title(f'STL Trend Extraction: {target_col}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
except Exception as e:
    print(f"STL failed: {e}")
"""
)

trend_basic_stacked = Algorithm(
    id="trend_basic_stacked",
    name="基础趋势绘制（分栏）",
    category="trend_plot",
    prompt="请按照原始样式对 {VAR_NAME} 进行趋势绘制（分栏布局）。每个数值列单独占一行子图，统一时间轴。实现要点：\\n1) 推断采样频率并重采样为统一时间轴（如 '1S'）；\\n2) 仅对数值列绘图，数据量大时先降采样（如 1S/5S 或 rolling）；\\n3) 使用 matplotlib，添加中文标题、轴标签、网格与图例；\\n4) 若 {VAR_NAME} 为 Series，则直接在单个子图绘制。",
    parameters=[],
    imports=["import matplotlib.pyplot as plt"],
    template="""
# Stacked Plot for {VAR_NAME}
df_plot = {VAR_NAME}.select_dtypes(include=['number'])
cols = df_plot.columns

fig, axes = plt.subplots(len(cols), 1, figsize=(12, 3*len(cols)), sharex=True)
if len(cols) == 1: axes = [axes]

for i, col in enumerate(cols):
    axes[i].plot(df_plot.index, df_plot[col])
    axes[i].set_title(col)
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
"""
)

trend_basic_overlay = Algorithm(
    id="trend_basic_overlay",
    name="基础趋势绘制（叠加）",
    category="trend_plot",
    prompt="请按照原始样式对 {VAR_NAME} 进行趋势绘制（叠加布局）。所有数值列绘制在同一坐标轴上并区分图例，统一时间轴。实现要点：\\n1) 推断采样频率并重采样为统一时间轴（如 '1S'）；\\n2) 仅对数值列绘图，数据量大时先降采样（如 1S/5S 或 rolling）；\\n3) 使用 matplotlib，添加中文标题、轴标签、网格与图例；\\n4) 若 {VAR_NAME} 为 Series，则直接在单图叠加绘制（仅一条曲线）。",
    parameters=[],
    imports=["import matplotlib.pyplot as plt"],
    template="""
# Overlay Plot for {VAR_NAME}
df_plot = {VAR_NAME}.select_dtypes(include=['number'])

plt.figure(figsize=(14, 7))
for col in df_plot.columns:
    plt.plot(df_plot.index, df_plot[col], label=col, alpha=0.7)

plt.title(f'Overlay Trend: {VAR_NAME}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
"""
)

trend_basic_grid = Algorithm(
    id="trend_basic_grid",
    name="基础趋势绘制（网格）",
    category="trend_plot",
    prompt="请按照原始样式对 {VAR_NAME} 进行趋势绘制（网格布局）。根据列数量自动计算行列数形成子图网格（如 2xN 或近似方阵），统一时间轴。实现要点：\\n1) 推断采样频率并重采样为统一时间轴（如 '1S'）；\\n2) 仅对数值列绘图，数据量大时先降采样（如 1S/5S 或 rolling）；\\n3) 使用 matplotlib，添加中文标题、轴标签、网格与图例；\\n4) 若 {VAR_NAME} 为 Series，则在单个子图绘制。",
    parameters=[],
    imports=["import matplotlib.pyplot as plt", "import math"],
    template="""
# Grid Plot for {VAR_NAME}
df_plot = {VAR_NAME}.select_dtypes(include=['number'])
cols = df_plot.columns
n_cols = len(cols)

# Calculate grid size
n_rows_grid = math.ceil(math.sqrt(n_cols))
n_cols_grid = math.ceil(n_cols / n_rows_grid)

fig, axes = plt.subplots(n_rows_grid, n_cols_grid, figsize=(15, 5*n_rows_grid))
axes = axes.flatten()

for i in range(len(axes)):
    if i < n_cols:
        axes[i].plot(df_plot.index, df_plot[cols[i]])
        axes[i].set_title(cols[i])
        axes[i].grid(True, alpha=0.3)
    else:
        axes[i].axis('off')

plt.tight_layout()
plt.show()
"""
)

algorithms = [
    trend_plot, trend_ma, trend_ewma, trend_loess, trend_polyfit,
    trend_stl_trend, trend_basic_stacked, trend_basic_overlay, trend_basic_grid
]
