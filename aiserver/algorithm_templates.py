"""
Algorithm Code Templates
Contains code templates for various data analysis algorithms.
These templates are designed to be inserted into Jupyter Notebook cells.
"""

from .algorithm_prompts import ALGORITHM_PROMPTS

ALGORITHM_TEMPLATES = {
    # --- Data Preprocessing ---
    "smoothing_sg": """
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

# Savitzky-Golay Smoothing for {VAR_NAME}
# Note: window_length must be odd and greater than polyorder
df_sg = {VAR_NAME}.copy()
numeric_cols = df_sg.select_dtypes(include=[np.number]).columns
window_length = 11  # Adjust based on data frequency
polyorder = 3

for col in numeric_cols:
    try:
        # Handle missing values before smoothing
        series = df_sg[col].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
        df_sg[f'{col}_sg'] = savgol_filter(series, window_length, polyorder)
    except Exception as e:
        print(f"Could not smooth column {col}: {e}")

# Display first few rows
df_sg.head()
""",

    "smoothing_ma": """
import pandas as pd
import numpy as np

# Moving Average Smoothing for {VAR_NAME}
df_ma = {VAR_NAME}.copy()
numeric_cols = df_ma.select_dtypes(include=[np.number]).columns
window_size = 5  # Adjust window size as needed

for col in numeric_cols:
    df_ma[f'{col}_ma'] = df_ma[col].rolling(window=window_size, center=True).mean()

# Display first few rows
df_ma.head()
""",

    "interpolation_time": """
import pandas as pd

# Time-weighted Interpolation for {VAR_NAME}
# Requires a DatetimeIndex for 'time' method
df_interp = {VAR_NAME}.copy()

# Ensure index is datetime if possible, otherwise use index
if not isinstance(df_interp.index, pd.DatetimeIndex):
    print("Warning: Index is not DatetimeIndex. Using linear interpolation instead of time-weighted.")
    method = 'linear'
else:
    method = 'time'

df_interp = df_interp.interpolate(method=method)

# Check remaining missing values
print("Remaining missing values:\\n", df_interp.isnull().sum())
df_interp.head()
""",

    "interpolation_spline": """
import pandas as pd
import numpy as np

# Spline Interpolation for {VAR_NAME}
df_spline = {VAR_NAME}.copy()

# Requires numeric index (or datetime converted to numeric) for spline
# Usually works best if we reset index to use integer index for interpolation if time index fails
# Here we try direct interpolation
try:
    df_spline = df_spline.interpolate(method='spline', order=3)
except Exception as e:
    print(f"Spline interpolation failed (index might not be compatible): {e}")
    print("Falling back to linear interpolation")
    df_spline = df_spline.interpolate(method='linear')

df_spline.head()
""",

    "resampling_down": """
import pandas as pd

# Downsampling (Aggregation) for {VAR_NAME}
# Requires DatetimeIndex
df_resampled = {VAR_NAME}.copy()

if isinstance(df_resampled.index, pd.DatetimeIndex):
    rule = '1H'  # Target frequency: 1 Hour, change to '1T' for 1 min, '1D' for 1 day
    
    # Define aggregation dictionary: mean for numeric, first/mode for others
    agg_dict = {}
    for col in df_resampled.columns:
        if pd.api.types.is_numeric_dtype(df_resampled[col]):
            agg_dict[col] = 'mean'
        else:
            agg_dict[col] = 'first'
            
    df_resampled = df_resampled.resample(rule).agg(agg_dict)
    print(f"Resampled to {rule} frequency. New shape: {df_resampled.shape}")
    display(df_resampled.head())
else:
    print("Error: {VAR_NAME} index is not a DatetimeIndex. Cannot resample.")
""",

    "alignment": """
import pandas as pd

# Multi-source Data Alignment using {VAR_NAME} as baseline
# This template assumes you have another dataframe to merge. 
# Please replace 'OTHER_DF' with your secondary dataframe name.

# Example placeholder for secondary dataframe (Remove if you have one)
# OTHER_DF = pd.DataFrame(...) 

baseline_df = {VAR_NAME}.sort_index()
# other_df = OTHER_DF.sort_index()

# Using merge_asof (requires sorted DatetimeIndex)
# df_aligned = pd.merge_asof(baseline_df, other_df, left_index=True, right_index=True, direction='nearest')

# print("Aligned DataFrame shape:", df_aligned.shape)
# df_aligned.head()
print("Please uncomment and set 'OTHER_DF' to perform alignment.")
""",

    # --- Exploratory Data Analysis (EDA) ---
    "summary_stats": """
import pandas as pd

# Time Series Summary Statistics for {VAR_NAME}
df_stats = {VAR_NAME}

print("--- Basic Statistics ---")
display(df_stats.describe())

print("\\n--- Skewness & Kurtosis ---")
numeric_df = df_stats.select_dtypes(include=['number'])
skew = numeric_df.skew()
kurt = numeric_df.kurtosis()
dist_df = pd.DataFrame({'Skewness': skew, 'Kurtosis': kurt})
display(dist_df)

print("\\n--- Time Range ---")
if isinstance(df_stats.index, pd.DatetimeIndex):
    print(f"Start: {df_stats.index.min()}")
    print(f"End:   {df_stats.index.max()}")
    print(f"Duration: {df_stats.index.max() - df_stats.index.min()}")
    
    # Estimate frequency
    diffs = df_stats.index.to_series().diff().dropna()
    print(f"Estimated Frequency (Median Diff): {diffs.median()}")
else:
    print("Index is not DatetimeIndex.")

print("\\n--- Missing Values ---")
missing = df_stats.isnull().sum()
missing_pct = (missing / len(df_stats)) * 100
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
display(missing_df[missing_df['Missing Count'] > 0])
""",

    "line_plot": """
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Multi-scale Time Series Plot for {VAR_NAME}
plt.figure(figsize=(15, 6))
sns.set_style("whitegrid")
# Support Chinese characters
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

df_plot = {VAR_NAME}.copy()
numeric_cols = df_plot.select_dtypes(include=['number']).columns

# Downsample for plotting if data is too large (>10k points)
if len(df_plot) > 10000:
    print(f"Data too large ({len(df_plot)} rows), downsampling for plot...")
    # Simple integer indexing downsample
    df_plot = df_plot.iloc[::len(df_plot)//5000]

for col in numeric_cols:
    plt.plot(df_plot.index, df_plot[col], label=col, alpha=0.8)

plt.title(f'{VAR_NAME} Time Series Plot', fontsize=14)
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
""",

    "spectral_analysis": """
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# Spectral Analysis (PSD) for {VAR_NAME}
df_psd = {VAR_NAME}.select_dtypes(include=['number'])
plt.figure(figsize=(12, 6))

for col in df_psd.columns:
    # Fill NA before spectral analysis
    data = df_psd[col].interpolate().fillna(method='bfill').values
    
    # Calculate PSD using Welch's method
    freqs, psd = welch(data)
    
    plt.semilogy(freqs, psd, label=col)

plt.title('Power Spectral Density (PSD)')
plt.xlabel('Frequency')
plt.ylabel('Power/Frequency (dB/Hz)')
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.show()
""",

    "autocorrelation": """
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# Autocorrelation (ACF) for {VAR_NAME}
df_acf = {VAR_NAME}.select_dtypes(include=['number'])

# Plot ACF for first few numeric columns
max_cols = 3
cols_to_plot = df_acf.columns[:max_cols]

fig, axes = plt.subplots(len(cols_to_plot), 1, figsize=(12, 4 * len(cols_to_plot)), sharex=False)
if len(cols_to_plot) == 1: axes = [axes]

for i, col in enumerate(cols_to_plot):
    # Drop NA
    data = df_acf[col].dropna()
    plot_acf(data, ax=axes[i], title=f'Autocorrelation: {col}', lags=50)

plt.tight_layout()
plt.show()
""",

    "decomposition": """
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
import pandas as pd

# STL Decomposition for {VAR_NAME}
# Requires DatetimeIndex with frequency inferred or set
df_stl = {VAR_NAME}.select_dtypes(include=['number']).copy()

# Try to set frequency if missing
if isinstance(df_stl.index, pd.DatetimeIndex) and df_stl.index.freq is None:
    inferred_freq = pd.infer_freq(df_stl.index)
    if inferred_freq:
        df_stl = df_stl.asfreq(inferred_freq)
        df_stl = df_stl.interpolate() # Fill gaps created by asfreq

target_col = df_stl.columns[0] # Decompose the first column
print(f"Decomposing column: {target_col}")

try:
    # Period is optional if freq is set, otherwise might need to specify
    res = STL(df_stl[target_col], robust=True).fit()
    
    fig = res.plot()
    fig.set_size_inches(12, 8)
    plt.show()
except Exception as e:
    print(f"STL Decomposition failed: {e}")
    print("Ensure data has a regular time frequency.")
""",

    "heatmap_distribution": """
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Time Series Heatmap for {VAR_NAME}
# Analyzes distribution by Date vs Hour
df_heat = {VAR_NAME}.copy()

if isinstance(df_heat.index, pd.DatetimeIndex):
    target_col = df_heat.select_dtypes(include=['number']).columns[0]
    
    df_heat['date'] = df_heat.index.date
    df_heat['hour'] = df_heat.index.hour
    
    pivot_table = df_heat.pivot_table(values=target_col, index='hour', columns='date', aggfunc='mean')
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot_table, cmap='viridis', robust=True)
    plt.title(f'Heatmap: {target_col} (Hour vs Date)')
    plt.ylabel('Hour of Day')
    plt.xlabel('Date')
    plt.show()
else:
    print("Index is not DatetimeIndex. Cannot generate time-based heatmap.")
""",

    # --- Anomaly Detection ---
    "threshold_sigma": """
import pandas as pd
import matplotlib.pyplot as plt

# 3-Sigma Anomaly Detection for {VAR_NAME}
df_anom = {VAR_NAME}.copy()
target_col = df_anom.select_dtypes(include=['number']).columns[0]
data = df_anom[target_col]

window = 20 # Rolling window size
rolling_mean = data.rolling(window=window).mean()
rolling_std = data.rolling(window=window).std()

# Define bounds
upper_bound = rolling_mean + 3 * rolling_std
lower_bound = rolling_mean - 3 * rolling_std

anomalies = data[(data > upper_bound) | (data < lower_bound)]

plt.figure(figsize=(15, 6))
plt.plot(data.index, data, label='Original', alpha=0.6)
plt.plot(rolling_mean.index, rolling_mean, 'k--', label='Moving Average', alpha=0.5)
plt.fill_between(rolling_mean.index, lower_bound, upper_bound, color='gray', alpha=0.2, label='3-Sigma Range')
plt.scatter(anomalies.index, anomalies, color='red', label='Anomaly', zorder=5)
plt.title(f'3-Sigma Anomaly Detection: {target_col}')
plt.legend()
plt.show()

print(f"Found {len(anomalies)} anomalies.")
""",

    "isolation_forest": """
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import pandas as pd

# Isolation Forest Anomaly Detection for {VAR_NAME}
df_iso = {VAR_NAME}.select_dtypes(include=['number']).dropna()

model = IsolationForest(contamination=0.05, random_state=42)
df_iso['anomaly'] = model.fit_predict(df_iso)

# -1 indicates anomaly, 1 indicates normal
anomalies = df_iso[df_iso['anomaly'] == -1]

target_col = df_iso.columns[0] # Visualize first column

plt.figure(figsize=(15, 6))
plt.plot(df_iso.index, df_iso[target_col], label='Normal', color='blue', alpha=0.6)
plt.scatter(anomalies.index, anomalies[target_col], color='red', label='Anomaly', s=20, zorder=5)
plt.title(f'Isolation Forest Detection: {target_col}')
plt.legend()
plt.show()
""",

    "change_point": """
import matplotlib.pyplot as plt
import ruptures as rpt
import numpy as np

# Change Point Detection for {VAR_NAME}
# Uses binary segmentation with L2 cost
df_cp = {VAR_NAME}.select_dtypes(include=['number']).dropna()
target_col = df_cp.columns[0]
signal = df_cp[target_col].values

# Detection
algo = rpt.Binseg(model="l2").fit(signal)
result = algo.predict(n_bkps=5) # Detect 5 change points

# Display
rpt.display(signal, result, figsize=(15, 6))
plt.title(f'Change Point Detection: {target_col}')
plt.show()
""",

    # --- Trend Plot ---
    "trend_ma": """
import matplotlib.pyplot as plt

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
""",

    "trend_ewma": """
import matplotlib.pyplot as plt

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
""",

    "trend_loess": """
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np

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
""",

    "trend_polyfit": """
import numpy as np
import matplotlib.pyplot as plt

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
""",

    "trend_stl_trend": """
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
import pandas as pd

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
""",

    "trend_basic_stacked": """
import matplotlib.pyplot as plt

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
""",

    "trend_basic_overlay": """
import matplotlib.pyplot as plt

# Overlay Plot for {VAR_NAME}
df_plot = {VAR_NAME}.select_dtypes(include=['number'])

plt.figure(figsize=(14, 7))
for col in df_plot.columns:
    plt.plot(df_plot.index, df_plot[col], label=col, alpha=0.7)

plt.title(f'Overlay Trend: {VAR_NAME}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
""",

    "trend_basic_grid": """
import matplotlib.pyplot as plt
import math

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
}

# 清理模板首行空行
CLEAN_TEMPLATES = {k: v.lstrip('\n') for k, v in ALGORITHM_TEMPLATES.items()}

ALGORITHM_DESCRIPTIONS = {
    "smoothing_sg": "对数据框的数值列执行 Savitzky-Golay 平滑：先插值填补缺失值，再按指定窗口长度和多项式阶数进行滤波，结果以 _sg 后缀写入新列。",
    "smoothing_ma": "对数值列使用居中窗口计算移动平均，得到平滑序列并以 _ma 后缀写入新列。",
    "interpolation_time": "依据索引类型选择插值方式：DatetimeIndex 使用 time 插值，否则使用 linear 插值，并输出剩余缺失统计。",
    "interpolation_spline": "尝试三次样条插值以获取更平滑的曲线，不兼容时回退到线性插值。",
    "resampling_down": "当索引为 DatetimeIndex 时按指定规则进行重采样聚合：数值列取均值，非数值列取 first。",
    "alignment": "以基准数据框为时间轴，使用 merge_asof 按最近时间点对齐其他数据源（示例需替换 OTHER_DF）。",
    "summary_stats": "输出基础统计 describe、偏度与峰度、时间范围与采样间隔估计、缺失值数量与占比的汇总。",
    "line_plot": "绘制所有数值列的时序曲线；数据量过大时先下采样；启用中文、网格、图例并自适配布局。",
    "spectral_analysis": "对数值列插值后使用 Welch 方法计算功率谱密度并以半对数坐标绘制，用于识别周期成分。",
    "autocorrelation": "对最多三个数值列计算并绘制自相关函数（ACF），默认滞后阶数为 50。",
    "decomposition": "对第一数值列进行 STL 分解，必要时推断频率并插值，展示趋势、季节项与残差。",
    "heatmap_distribution": "将 DatetimeIndex 拆分为日期与小时，按均值汇聚形成透视表并绘制热力图观察日内/季节性模式。",
    "threshold_sigma": "计算滚动均值与标准差，标记超过 mean±3*std 的异常点，并在图中高亮显示。",
    "isolation_forest": "对数值列训练孤立森林（默认污染率 0.05），输出模型判定的异常点并可视化。",
    "change_point": "对第一数值列使用 ruptures 的二分段 L2 模型检测变点，返回并展示切分位置。",
    "trend_ma": "对第一数值列按固定窗口计算移动平均作为趋势线，并与原序列对比绘制。",
    "trend_ewma": "对第一数值列计算指数加权移动平均作为趋势（span 可调），并与原序列对比绘制。",
    "trend_loess": "对第一数值列使用 LOWESS/LOESS 进行非参数平滑拟合，得到平滑趋势线。",
    "trend_polyfit": "对第一数值列拟合二次多项式得到趋势线，并与原序列对比展示。",
    "trend_stl_trend": "对第一数值列进行 STL 分解并提取 trend 分量，绘制趋势随时间的变化。",
    "trend_basic_stacked": "为每个数值列单独绘制子图并垂直堆叠展示，便于逐列观察。",
    "trend_basic_overlay": "将所有数值列叠加在同一坐标轴中展示，便于横向对比。",
    "trend_basic_grid": "将各数值列按网格布局分别绘制，自动计算网格大小以适配列数。",
}

def get_library_metadata():
    """返回前端算法库元数据，使用实现说明替代提示词描述。"""
    library = {}
    for cat_key, cat_data in ALGORITHM_PROMPTS.items():
        cat_label = cat_data["label"]
        library[cat_label] = []
        for algo in cat_data["algorithms"]:
            algo_id = algo["id"]
            template = CLEAN_TEMPLATES.get(algo_id, f"# Template for {algo['name']} not found.")
            description = ALGORITHM_DESCRIPTIONS.get(algo_id, "暂无实现说明")
            
            library[cat_label].append({
                "id": algo_id,
                "name": algo["name"],
                "description": description,
                "category": cat_label,
                "template": template,
                # Keep backward compatibility fields if needed, or dummy them
                "docstring": template, 
                "signature": "",
                "module": "algorithm_templates",
                "args": [] 
            })
            
    return library
