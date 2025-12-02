from .base import Algorithm, AlgorithmParameter, Port

trend_plot = Algorithm(
    id="trend_plot",
    name="通用趋势图 (Trend)",
    category="trend_plot",
    prompt="请根据配置绘制 {VAR_NAME} 的趋势图。支持自定义 X 轴、Y 轴列、标题、网格等设置。",
    parameters=[
        AlgorithmParameter(name="x_column", type="str", default="", label="X轴列名", description="作为X轴的列 (留空则使用索引)", widget="column-selector", priority="critical"),
        AlgorithmParameter(name="y_columns", type="list", default=[], label="Y轴列名", description="Y轴数据列 (留空则绘制所有数值列)", widget="column-selector", priority="critical"),
        AlgorithmParameter(name="plot_type", type="str", default="叠加显示", label="绘图方式", description="选择绘图方式: 叠加显示、堆叠显示、分栏显示或网格显示", widget="select", options=["叠加显示", "堆叠显示", "分栏显示", "网格显示"], priority="critical"),
        AlgorithmParameter(name="title", type="str", default="趋势图", label="图表标题", description="图表的标题", priority="non-critical"),
        AlgorithmParameter(name="xlabel", type="str", default="", label="X轴标签", description="X轴的显示标签", priority="non-critical"),
        AlgorithmParameter(name="ylabel", type="str", default="", label="Y轴标签", description="Y轴的显示标签", priority="non-critical"),
        AlgorithmParameter(name="grid", type="bool", default=True, label="显示网格", description="是否显示背景网格", priority="non-critical"),
        AlgorithmParameter(name="figsize", type="str", default="", label="图像尺寸", description="图像大小元组，例如 (10, 6)。留空则自动适应CELL宽度", priority="non-critical")
    ],
    imports=["import matplotlib.pyplot as plt", "import pandas as pd", "import math"],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    template="""
# Trend Plot (Complex)
# Input: {VAR_NAME}
# Output: {OUTPUT_VAR} (Pass-through)

import matplotlib.pyplot as plt
import pandas as pd
import math

{OUTPUT_VAR} = {VAR_NAME}

try:
    # Configuration
    x_col = '{x_column}'
    y_cols = {y_columns}
    title = '{title}'
    xlabel = '{xlabel}'
    ylabel = '{ylabel}'
    show_grid = {grid}
    figsize_str = '{figsize}'
    plot_type = '{plot_type}'
    
    # Parse figsize
    try:
        if figsize_str:
            figsize = eval(figsize_str)
        else:
            # Auto-adjust figsize based on plot type for better cell width fitting
            if plot_type_en == 'overlay' or plot_type_en == 'stacked':
                # Wider default for single plot to fit cell width
                figsize = (15, 6)
            elif plot_type_en == 'subplots':
                # Taller default for subplots, width fits cell
                figsize = (15, 3*len(y_cols))
            elif plot_type_en == 'grid':
                # Grid layout with auto-calculated size
                n_rows = math.ceil(math.sqrt(len(y_cols)))
                n_cols_grid = math.ceil(len(y_cols) / n_rows)
                figsize = (18, 4*n_rows)  # Wider to fit cell width
            else:
                # Fallback default
                figsize = (15, 6)
    except:
        # Fallback if parsing fails
        figsize = (15, 6)

    # Convert Chinese plot_type to English for code logic
    plot_type_en = plot_type
    if plot_type == '叠加显示':
        plot_type_en = 'overlay'
    elif plot_type == '堆叠显示':
        plot_type_en = 'stacked'
    elif plot_type == '分栏显示':
        plot_type_en = 'subplots'
    elif plot_type == '网格显示':
        plot_type_en = 'grid'

    # Determine if the DataFrame is a time series
    is_time_series = False
    x_data = None
    
    # Check if index is a DatetimeIndex (most common case for time series)
    if isinstance({VAR_NAME}.index, pd.DatetimeIndex):
        is_time_series = True
        x_data = {VAR_NAME}.index
    
    # If x_column is specified, use it instead of index
    if x_col and x_col in {VAR_NAME}.columns:
        # Check if the specified column is datetime-like
        if pd.api.types.is_datetime64_any_dtype({VAR_NAME}[x_col]):
            is_time_series = True
            x_data = {VAR_NAME}[x_col]
        else:
            # Try to convert to datetime if it's not already
            try:
                x_data = pd.to_datetime({VAR_NAME}[x_col])
                is_time_series = True
                print(f"Converted column '{x_col}' to datetime")
            except Exception:
                print(f"Warning: Could not convert column '{x_col}' to datetime. Using original values.")
                x_data = {VAR_NAME}[x_col]
    elif x_col:
        # x_column specified but not found, use index
        print(f"Warning: X column '{x_col}' not found, using index.")
        if isinstance({VAR_NAME}.index, pd.DatetimeIndex):
            is_time_series = True
        x_data = {VAR_NAME}.index
    
    # If x_data is still None (shouldn't happen), use index
    if x_data is None:
        x_data = {VAR_NAME}.index
        if isinstance(x_data, pd.DatetimeIndex):
            is_time_series = True

    if not y_cols:
        # If empty (or parsed as empty), fallback to numeric columns only
        y_cols = {VAR_NAME}.select_dtypes(include=['number']).columns.tolist()
    
    # Remove x_col from y_cols if present to avoid plotting time/index on Y-axis
    if x_col and x_col in y_cols:
        y_cols.remove(x_col)

    # Support Chinese characters in title and labels if needed
    plt.rcParams['font.sans-serif'] = ['SimHei'] # Use SimHei for Chinese
    plt.rcParams['axes.unicode_minus'] = False   # Fix minus sign

    # Create plot based on plot_type_en
    if plot_type_en == 'overlay':
        # Overlay plot (default) - all lines on same axes
        plt.figure(figsize=figsize)
        for col in y_cols:
            if col in {VAR_NAME}.columns:
                plt.plot(x_data, {VAR_NAME}[col], label=col)
            else:
                print(f"Warning: Y column '{col}' not found.")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(show_grid)
        plt.legend()
        plt.tight_layout()
    
    elif plot_type_en == 'stacked':
        # Stacked plot - lines stacked on top of each other
        plt.figure(figsize=figsize)
        # Calculate cumulative sum for stacking
        df_stacked = {VAR_NAME}[y_cols].cumsum(axis=1)
        for i, col in enumerate(y_cols):
            if col in {VAR_NAME}.columns:
                if i == 0:
                    # First column is plotted from 0
                    plt.fill_between(x_data, 0, df_stacked[col], label=col, alpha=0.7)
                else:
                    # Subsequent columns are plotted from previous cumulative sum
                    plt.fill_between(x_data, df_stacked[y_cols[i-1]], df_stacked[col], label=col, alpha=0.7)
            else:
                print(f"Warning: Y column '{col}' not found.")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(show_grid)
        plt.legend()
        plt.tight_layout()
    
    elif plot_type_en == 'subplots':
        # Subplots - each line in its own subplot, stacked vertically
        n_cols = len(y_cols)
        fig, axes = plt.subplots(n_cols, 1, figsize=(figsize[0], 3*n_cols), sharex=True)
        if n_cols == 1:
            axes = [axes]
        
        for i, col in enumerate(y_cols):
            if col in {VAR_NAME}.columns:
                axes[i].plot(x_data, {VAR_NAME}[col], label=col)
                axes[i].set_title(col)
                axes[i].grid(show_grid)
                axes[i].legend()
            else:
                print(f"Warning: Y column '{col}' not found.")
        
        plt.suptitle(title, y=0.99, fontsize=16)
        plt.xlabel(xlabel)
        plt.tight_layout()
    
    elif plot_type_en == 'grid':
        # Grid plot - each line in its own subplot, arranged in a grid
        n_cols = len(y_cols)
        if n_cols == 0:
            print("No Y columns selected for plotting.")
        else:
            # Calculate grid size
            n_rows = math.ceil(math.sqrt(n_cols))
            n_cols_grid = math.ceil(n_cols / n_rows)
            
            fig, axes = plt.subplots(n_rows, n_cols_grid, figsize=(figsize[0]*n_cols_grid/2, figsize[1]*n_rows/2))
            axes = axes.flatten()
            
            for i in range(len(axes)):
                if i < n_cols:
                    col = y_cols[i]
                    if col in {VAR_NAME}.columns:
                        axes[i].plot(x_data, {VAR_NAME}[col], label=col)
                        axes[i].set_title(col)
                        axes[i].grid(show_grid)
                        axes[i].legend()
                    else:
                        print(f"Warning: Y column '{col}' not found.")
                else:
                    # Hide unused subplots
                    axes[i].axis('off')
            
            plt.suptitle(title, y=0.99, fontsize=16)
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
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
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
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
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
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
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
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
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
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
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

# OHLC 重采样
trend_ohlc = Algorithm(
    id="trend_ohlc",
    name="OHLC重采样",
    category="trend_plot",
    prompt="请对{VAR_NAME} 进行OHLC重采样。将时间序列数据重采样为指定频率的开盘价(Open)、最高价(High)、最低价(Low)和收盘价(Close)，并绘制蜡烛图。",
    parameters=[
        AlgorithmParameter(name="y_columns", type="list", default=[], label="Y轴列名", description="要进行OHLC重采样的列 (留空则使用所有数值列)", widget="column-selector", priority="critical"),
        AlgorithmParameter(name="resample_freq", type="str", default="5T", label="重采样频率", description="重采样频率，如 1T=1分钟, 5T=5分钟, 1H=1小时, 1D=1天", priority="critical"),
        AlgorithmParameter(name="title", type="str", default="OHLC蜡烛图", label="图表标题", description="图表的标题", priority="non-critical"),
        AlgorithmParameter(name="figsize", type="str", default="(15, 8)", label="图像尺寸", description="图像大小元组，例如 (15, 8)", priority="non-critical")
    ],
    imports=["import pandas as pd", "import matplotlib.pyplot as plt", "import mplfinance.original_flavor as mpf"],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    template="""
# OHLC Resampling for {VAR_NAME}
{OUTPUT_VAR} = {VAR_NAME}.copy()

# Get parameters
y_columns = {y_columns}
resample_freq = '{resample_freq}'
title = '{title}'
figsize_str = '{figsize}'

# Parse figsize
try:
    figsize = eval(figsize_str)
except:
    figsize = (15, 8)

# Determine Y columns
if not y_columns:
    # Use all numeric columns if none specified
    y_columns = {OUTPUT_VAR}.select_dtypes(include=['number']).columns.tolist()

# Check if the index is a DatetimeIndex
if not isinstance({OUTPUT_VAR}.index, pd.DatetimeIndex):
    print("错误: 数据索引不是时间索引，无法进行OHLC重采样。")
else:
    # Filter to selected columns
    ohlc_data = {OUTPUT_VAR}[y_columns].copy()
    
    # Perform OHLC resampling for each column
    for col in y_columns:
        # Resample to OHLC
        ohlc = ohlc_data[col].resample(resample_freq).ohlc()
        
        # Plot OHLC chart using mplfinance
        plt.figure(figsize=figsize)
        mpf.candlestick2_ochl(plt.gca(), ohlc['open'], ohlc['high'], ohlc['low'], ohlc['close'], width=0.6, colorup='green', colordown='red', alpha=0.8)
        
        plt.title(f"OHLC Chart for {col} (Resampled to {resample_freq})")
        plt.xlabel('时间')
        plt.ylabel(col)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
"""
)

# 数据包络线绘制
trend_envelope = Algorithm(
    id="trend_envelope",
    name="数据包络线绘制",
    category="trend_plot",
    prompt="请对{VAR_NAME} 绘制数据包络线。使用滚动窗口的最大值和最小值计算上、下包络线，并与原始曲线一起绘制。",
    parameters=[
        AlgorithmParameter(name="y_columns", type="list", default=[], label="Y轴列名", description="要绘制包络线的列 (留空则使用所有数值列)", widget="column-selector", priority="critical"),
        AlgorithmParameter(name="window_duration", type="str", default="1min", label="窗口时长", description="时间窗口长度，如 1min=1分钟, 5min=5分钟, 1h=1小时", widget="select", options=["30s", "1min", "5min", "15min", "30min", "1h", "2h", "6h", "12h", "1D"], priority="critical"),
        AlgorithmParameter(name="title", type="str", default="数据包络线图", label="图表标题", description="图表的标题", priority="non-critical"),
        AlgorithmParameter(name="figsize", type="str", default="(15, 8)", label="图像尺寸", description="图像大小元组，例如 (15, 8)", priority="non-critical")
    ],
    imports=["import pandas as pd", "import matplotlib.pyplot as plt", "import numpy as np"],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    template="""
# Data Envelope Plot for {VAR_NAME}
{OUTPUT_VAR} = {VAR_NAME}.copy()

# Get parameters
y_columns = {y_columns}
window_duration = '{window_duration}'
title = '{title}'
figsize_str = '{figsize}'

# Parse figsize
try:
    figsize = eval(figsize_str)
except:
    figsize = (15, 8)

# Set Chinese font support
plt.rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei for Chinese
plt.rcParams['axes.unicode_minus'] = False   # Fix minus sign display

# Determine Y columns
if not y_columns:
    # Use all numeric columns if none specified
    y_columns = {OUTPUT_VAR}.select_dtypes(include=['number']).columns.tolist()

# Filter to selected columns
envelope_data = {OUTPUT_VAR}[y_columns].copy()

# Check if the index is a DatetimeIndex
is_time_series = isinstance(envelope_data.index, pd.DatetimeIndex)

# Calculate window size based on time duration
window_size = 60  # Default window size if not time series
if is_time_series:
    # Calculate time difference between consecutive points
    time_diff = envelope_data.index.to_series().diff().median()
    if pd.isna(time_diff):
        time_diff = pd.Timedelta(seconds=1)  # Default to 1 second if no valid difference
    
    # Convert window duration to Timedelta
    window_timedelta = pd.Timedelta(window_duration)
    
    # Calculate window size in points
    window_size = int(window_timedelta / time_diff)
    
    # Ensure minimum window size
    window_size = max(5, window_size)

# Plot envelope for each column
for col in y_columns:
    # Calculate upper and lower envelopes
    upper_envelope = envelope_data[col].rolling(window=window_size, center=True).max()
    lower_envelope = envelope_data[col].rolling(window=window_size, center=True).min()
    
    # Create plot
    plt.figure(figsize=figsize)
    plt.plot(envelope_data.index, envelope_data[col], label='原始数据', alpha=0.7)
    plt.plot(envelope_data.index, upper_envelope, label=f'上包络线 (窗口={window_duration})', color='red', linestyle='--')
    plt.plot(envelope_data.index, lower_envelope, label=f'下包络线 (窗口={window_duration})', color='green', linestyle='--')
    
    plt.title(f"数据包络线: {col}")
    plt.xlabel('时间' if is_time_series else '索引')
    plt.ylabel(col)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
"""
)

algorithms = [
    trend_plot, trend_ma, trend_ewma, trend_loess, trend_polyfit,
    trend_stl_trend, trend_ohlc, trend_envelope
]
