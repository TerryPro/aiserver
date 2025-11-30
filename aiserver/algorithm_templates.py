"""
Algorithm Code Templates
Contains code templates for various data analysis algorithms.
These templates are designed to be inserted into Jupyter Notebook cells.
"""

from .algorithm_prompts import ALGORITHM_PROMPTS
from .algorithm_registry import ALGORITHM_PARAMETERS

ALGORITHM_IMPORTS = {
    "load_csv": ["import pandas as pd", "import os"],
    "plot_custom": ["import matplotlib.pyplot as plt"],
    "smoothing_sg": ["import pandas as pd", "import numpy as np", "from scipy.signal import savgol_filter"],
    "smoothing_ma": ["import pandas as pd", "import numpy as np"],
    "interpolation_time": ["import pandas as pd"],
    "interpolation_spline": ["import pandas as pd", "import numpy as np"],
    "resampling_down": ["import pandas as pd"],
    "alignment": ["import pandas as pd"],
    "feature_scaling": ["import pandas as pd", "from sklearn.preprocessing import StandardScaler, MinMaxScaler"],
    "diff_transform": ["import pandas as pd", "import matplotlib.pyplot as plt"],
    "outlier_clip": ["import pandas as pd"],
    "feature_extraction_time": ["import pandas as pd"],
    "feature_lag": ["import pandas as pd"],
    "transform_log": ["import pandas as pd", "import numpy as np", "import matplotlib.pyplot as plt"],
    "filter_butterworth": ["import pandas as pd", "import numpy as np", "from scipy.signal import butter, filtfilt", "import matplotlib.pyplot as plt"],
    "summary_stats": ["import pandas as pd"],
    "line_plot": ["import matplotlib.pyplot as plt", "import seaborn as sns", "import pandas as pd"],
    "spectral_analysis": ["import numpy as np", "import matplotlib.pyplot as plt", "from scipy.signal import welch"],
    "autocorrelation": ["import matplotlib.pyplot as plt", "from statsmodels.graphics.tsaplots import plot_acf"],
    "decomposition": ["import matplotlib.pyplot as plt", "from statsmodels.tsa.seasonal import STL", "import pandas as pd"],
    "heatmap_distribution": ["import seaborn as sns", "import matplotlib.pyplot as plt", "import pandas as pd"],
    "threshold_sigma": ["import pandas as pd", "import matplotlib.pyplot as plt"],
    "isolation_forest": ["from sklearn.ensemble import IsolationForest", "import matplotlib.pyplot as plt", "import pandas as pd"],
    "change_point": ["import matplotlib.pyplot as plt", "import ruptures as rpt", "import numpy as np"],
    "trend_ma": ["import matplotlib.pyplot as plt"],
    "trend_ewma": ["import matplotlib.pyplot as plt"],
    "trend_loess": ["import statsmodels.api as sm", "import matplotlib.pyplot as plt", "import numpy as np"],
    "trend_polyfit": ["import numpy as np", "import matplotlib.pyplot as plt"],
    "trend_stl_trend": ["from statsmodels.tsa.seasonal import STL", "import matplotlib.pyplot as plt", "import pandas as pd"],
    "trend_basic_stacked": ["import matplotlib.pyplot as plt"],
    "trend_basic_overlay": ["import matplotlib.pyplot as plt"],
    "trend_basic_grid": ["import matplotlib.pyplot as plt", "import math"],
    "merge_dfs": ["import pandas as pd"],
    "train_test_split": ["from sklearn.model_selection import train_test_split", "import pandas as pd"],
    "import_variable": ["import pandas as pd"],
    "trend_plot": ["import matplotlib.pyplot as plt"],
}

ALGORITHM_TEMPLATES = {
    # --- Data Loading ---
    "load_csv": """
# Load CSV Data
filepath = '{filepath}'
if not os.path.exists(filepath):
    print(f"Error: File not found at {{filepath}}")
else:
    {OUTPUT_VAR} = pd.read_csv(filepath)
    print(f"Loaded {OUTPUT_VAR} with shape: {{{OUTPUT_VAR}.shape}}")
    display({OUTPUT_VAR}.head())
""",

    "import_variable": """
# Import Existing Variable
# Source: {variable_name}
# Output: {OUTPUT_VAR}

try:
    if '{variable_name}' not in locals():
        print(f"Error: Variable '{{variable_name}}' not found in current session.")
    else:
        # Create a copy to avoid modifying the original variable accidentally
        {OUTPUT_VAR} = {variable_name}.copy()
        print(f"Imported '{{variable_name}}' as '{OUTPUT_VAR}' with shape: {{{OUTPUT_VAR}.shape}}")
        display({OUTPUT_VAR}.head())
except Exception as e:
    print(f"Import failed: {e}")
""",

    "trend_plot": """
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
            print(f"Warning: Could not convert column '{{x_col}}' to datetime. Using original values.")
            x_data = {VAR_NAME}[x_col]
    else:
        x_data = {VAR_NAME}.index
        if x_col:
            print(f"Warning: X column '{{x_col}}' not found, using index.")

    # Parse Y columns
    if y_cols_str:
        y_cols = [c.strip() for c in y_cols_str.split(',') if c.strip()]
    else:
        y_cols = []

    if not y_cols:
        # If no Y columns specified, plot all numeric columns
        y_cols = {VAR_NAME}.select_dtypes(include=['number']).columns.tolist()

    for col in y_cols:
        if col in {VAR_NAME}.columns:
            plt.plot(x_data, {VAR_NAME}[col], label=col)
        else:
            print(f"Warning: Y column '{{col}}' not found.")

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
""",

    "merge_dfs": """
# Merge DataFrames
# Inputs: {left}, {right}
# Output: {merged}

try:
    # Check if inputs are available
    if '{left}' not in locals() or '{right}' not in locals():
        print("Error: Input DataFrames not found.")
    else:
        on_col = '{on}'
        if on_col == '':
            # Merge on index if no column specified
            {merged} = pd.merge({left}, {right}, how='{how}', left_index=True, right_index=True)
        else:
            {merged} = pd.merge({left}, {right}, how='{how}', on=on_col)
            
        print(f"Merged shape: {{{merged}.shape}}")
        display({merged}.head())
except Exception as e:
    print(f"Merge failed: {e}")
""",

    "train_test_split": """
# Train/Test Split
# Input: {data}
# Outputs: {X_train}, {X_test}, {y_train}, {y_test}

try:
    target = '{target_column}'
    if target not in {data}.columns:
        print(f"Error: Target column '{{target}}' not found in DataFrame.")
    else:
        X = {data}.drop(columns=[target])
        y = {data}[target]
        
        {X_train}, {X_test}, {y_train}, {y_test} = train_test_split(
            X, y, test_size={test_size}, random_state={random_state}
        )
        
        print(f"Train shape: X={{{X_train}.shape}}, y={{{y_train}.shape}}")
        print(f"Test shape:  X={{{X_test}.shape}},  y={{{y_test}.shape}}")
except Exception as e:
    print(f"Split failed: {e}")
""",

    # --- Visualization ---
    "plot_custom": """
# Generic Plot
# Plot type: {plot_type}, Column: {column}
if '{column}':
    {VAR_NAME}.plot(kind='{plot_type}', y='{column}')
else:
    {VAR_NAME}.plot(kind='{plot_type}')
plt.show()
""",

    # --- Data Preprocessing ---
    # 对数据框的数值列执行 Savitzky-Golay 平滑：先插值填补缺失值，再按指定窗口长度和多项式阶数进行滤波，结果以 _sg 后缀写入新列。
    "smoothing_sg": """
# Savitzky-Golay Smoothing for {VAR_NAME}
# Note: window_length must be odd and greater than polyorder
{OUTPUT_VAR} = {VAR_NAME}.copy()
numeric_cols = {OUTPUT_VAR}.select_dtypes(include=[np.number]).columns
window_length = {window_length}  # Adjust based on data frequency
polyorder = {polyorder}

for col in numeric_cols:
    try:
        # Handle missing values before smoothing
        series = {OUTPUT_VAR}[col].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
        {OUTPUT_VAR}[f'{col}_sg'] = savgol_filter(series, window_length, polyorder)
    except Exception as e:
        print(f"Could not smooth column {col}: {e}")

# Display first few rows
{OUTPUT_VAR}.head()
""",

    # 对数值列使用居中窗口计算移动平均，得到平滑序列并以 _ma 后缀写入新列。
    "smoothing_ma": """
# Moving Average Smoothing for {VAR_NAME}
{OUTPUT_VAR} = {VAR_NAME}.copy()
numeric_cols = {OUTPUT_VAR}.select_dtypes(include=[np.number]).columns
window_size = {window_size}  # Adjust window size as needed

for col in numeric_cols:
    {OUTPUT_VAR}[f'{col}_ma'] = {OUTPUT_VAR}[col].rolling(window=window_size, center=True).mean()

# Display first few rows
{OUTPUT_VAR}.head()
""",

    # 依据索引类型选择插值方式：DatetimeIndex 使用 time 插值，否则使用 linear 插值，并输出剩余缺失统计。
    "interpolation_time": """
# Time-weighted Interpolation for {VAR_NAME}
# Requires a DatetimeIndex for 'time' method
{OUTPUT_VAR} = {VAR_NAME}.copy()

# Ensure index is datetime if possible, otherwise use index
if not isinstance({OUTPUT_VAR}.index, pd.DatetimeIndex):
    print("Warning: Index is not DatetimeIndex. Using linear interpolation instead of time-weighted.")
    method = 'linear'
else:
    method = 'time'

{OUTPUT_VAR} = {OUTPUT_VAR}.interpolate(method=method)

# Check remaining missing values
print("Remaining missing values:\\n", {OUTPUT_VAR}.isnull().sum())
{OUTPUT_VAR}.head()
""",

    # 尝试三次样条插值以获取更平滑的曲线，不兼容时回退到线性插值。
    "interpolation_spline": """
# Spline Interpolation for {VAR_NAME}
{OUTPUT_VAR} = {VAR_NAME}.copy()

# Requires numeric index (or datetime converted to numeric) for spline
# Usually works best if we reset index to use integer index for interpolation if time index fails
# Here we try direct interpolation
try:
    {OUTPUT_VAR} = {OUTPUT_VAR}.interpolate(method='spline', order={order})
except Exception as e:
    print(f"Spline interpolation failed (index might not be compatible): {e}")
    print("Falling back to linear interpolation")
    {OUTPUT_VAR} = {OUTPUT_VAR}.interpolate(method='linear')

{OUTPUT_VAR}.head()
""",

    # 当索引为 DatetimeIndex 时按指定规则进行重采样聚合：数值列取均值，非数值列取 first。
    "resampling_down": """
# Downsampling (Aggregation) for {VAR_NAME}
# Requires DatetimeIndex
{OUTPUT_VAR} = {VAR_NAME}.copy()

if isinstance({OUTPUT_VAR}.index, pd.DatetimeIndex):
    rule = '{rule}'  # Target frequency: 1 Hour, change to '1T' for 1 min, '1D' for 1 day
    
    # Define aggregation dictionary: mean for numeric, first/mode for others
    agg_dict = {}
    for col in {OUTPUT_VAR}.columns:
        if pd.api.types.is_numeric_dtype({OUTPUT_VAR}[col]):
            agg_dict[col] = 'mean'
        else:
            agg_dict[col] = 'first'
            
    {OUTPUT_VAR} = {OUTPUT_VAR}.resample(rule).agg(agg_dict)
    print(f"Resampled to {rule} frequency. New shape: {{{OUTPUT_VAR}.shape}}")
    display({OUTPUT_VAR}.head())
else:
    print("Error: {VAR_NAME} index is not a DatetimeIndex. Cannot resample.")
""",

    # 以基准数据框为时间轴，使用 merge_asof 按最近时间点对齐其他数据源（示例需替换 OTHER_DF）。
    "alignment": """
# Multi-source Data Alignment using {VAR_NAME} as baseline
# This template assumes you have another dataframe to merge. 
# Please replace 'OTHER_DF' with your secondary dataframe name.

# Example placeholder for secondary dataframe (Remove if you have one)
# OTHER_DF = pd.DataFrame(...) 

baseline_df = {VAR_NAME}.sort_index()
# other_df = OTHER_DF.sort_index()

# Using merge_asof (requires sorted DatetimeIndex)
# {OUTPUT_VAR} = pd.merge_asof(baseline_df, other_df, left_index=True, right_index=True, direction='nearest')

# print("Aligned DataFrame shape:", {OUTPUT_VAR}.shape)
# {OUTPUT_VAR}.head()
print("Please uncomment and set 'OTHER_DF' to perform alignment.")
""",

    # 对数值列进行 Z-Score 标准化和 Min-Max 归一化，生成对应后缀的新列。
    "feature_scaling": """
# Feature Scaling for {VAR_NAME}
{OUTPUT_VAR} = {VAR_NAME}.select_dtypes(include=['number']).copy()
cols = {OUTPUT_VAR}.columns

# 1. Z-Score Standardization
scaler_std = StandardScaler()
df_std = pd.DataFrame(scaler_std.fit_transform({OUTPUT_VAR}), index={OUTPUT_VAR}.index, columns=[f"{c}_std" for c in cols])

# 2. Min-Max Normalization
scaler_minmax = MinMaxScaler()
df_minmax = pd.DataFrame(scaler_minmax.fit_transform({OUTPUT_VAR}), index={OUTPUT_VAR}.index, columns=[f"{c}_minmax" for c in cols])

# Combine results
{OUTPUT_VAR} = pd.concat([{OUTPUT_VAR}, df_std, df_minmax], axis=1)
print("Scaling completed. Added _std and _minmax columns.")
display({OUTPUT_VAR}.head())
""",

    # 计算一阶和二阶差分以消除趋势，并绘制差分后的时序对比图。
    "diff_transform": """
# Difference Transform for {VAR_NAME}
{OUTPUT_VAR} = {VAR_NAME}.select_dtypes(include=['number']).copy()

# 1st Order Difference
df_diff_1 = {OUTPUT_VAR}.diff(1)
# 2nd Order Difference
df_diff_2 = {OUTPUT_VAR}.diff(2)

# Plotting
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
{OUTPUT_VAR}.plot(ax=axes[0], title="Original Data")
df_diff_1.plot(ax=axes[1], title="1st Order Difference")
df_diff_2.plot(ax=axes[2], title="2nd Order Difference")

plt.tight_layout()
plt.show()
""",

    # 将数值列中超过 1% 和 99% 分位数的极端值进行盖帽（截断）处理。
    "outlier_clip": """
# Outlier Winsorization for {VAR_NAME}
{OUTPUT_VAR} = {VAR_NAME}.copy()
numeric_cols = {OUTPUT_VAR}.select_dtypes(include=['number']).columns

for col in numeric_cols:
    # Calculate bounds
    lower = {OUTPUT_VAR}[col].quantile(0.01)
    upper = {OUTPUT_VAR}[col].quantile(0.99)
    
    # Clip
    {OUTPUT_VAR}[f'{col}_clipped'] = {OUTPUT_VAR}[col].clip(lower=lower, upper=upper)
    
    print(f"Column {col}: Clipped to [{lower:.4f}, {upper:.4f}]")

display({OUTPUT_VAR}.head())
""",

    # 从 DatetimeIndex 提取小时、星期、月份、是否周末等时间特征列。
    "feature_extraction_time": """
# Time Feature Extraction for {VAR_NAME}
{OUTPUT_VAR} = {VAR_NAME}.copy()

if isinstance({OUTPUT_VAR}.index, pd.DatetimeIndex):
    {OUTPUT_VAR}['hour'] = {OUTPUT_VAR}.index.hour
    {OUTPUT_VAR}['dayofweek'] = {OUTPUT_VAR}.index.dayofweek
    {OUTPUT_VAR}['month'] = {OUTPUT_VAR}.index.month
    {OUTPUT_VAR}['quarter'] = {OUTPUT_VAR}.index.quarter
    {OUTPUT_VAR}['is_weekend'] = {OUTPUT_VAR}.index.dayofweek.isin([5, 6]).astype(int)
    
    print("Time features added:")
    display({OUTPUT_VAR}[['hour', 'dayofweek', 'month', 'is_weekend']].head())
else:
    print("Error: Index is not DatetimeIndex.")
""",

    # 生成指定滞后步数（1-3）的特征列，用于构建自回归模型数据集。
    "feature_lag": """
# Lag Feature Generation for {VAR_NAME}
{OUTPUT_VAR} = {VAR_NAME}.select_dtypes(include=['number']).copy()
target_cols = {OUTPUT_VAR}.columns
lags = [1, 2, 3]

for col in target_cols:
    for lag in lags:
        {OUTPUT_VAR}[f'{col}_lag_{lag}'] = {OUTPUT_VAR}[col].shift(lag)

# Drop rows with NaNs created by shifting
{OUTPUT_VAR}.dropna(inplace=True)

print(f"Generated lag features for {lags}. New shape: {{{OUTPUT_VAR}.shape}}")
display({OUTPUT_VAR}.head())
""",

    # 对数值列进行 Log1p 变换以平滑分布，自动处理负值偏移，并绘制分布对比图。
    "transform_log": """
# Log Transformation for {VAR_NAME}
{OUTPUT_VAR} = {VAR_NAME}.select_dtypes(include=['number']).copy()
cols = {OUTPUT_VAR}.columns

fig, axes = plt.subplots(len(cols), 2, figsize=(12, 4*len(cols)))
if len(cols) == 1: axes = np.array([axes])

for i, col in enumerate(cols):
    # Use log1p to handle zeros, ensure non-negative
    if ({OUTPUT_VAR}[col] < 0).any():
        print(f"Warning: Column {col} contains negative values. Adding offset before log.")
        offset = abs({OUTPUT_VAR}[col].min()) + 1
        data_to_log = {OUTPUT_VAR}[col] + offset
    else:
        data_to_log = {OUTPUT_VAR}[col]
        
    {OUTPUT_VAR}[f'{col}_log'] = np.log1p(data_to_log)
    
    # Plot Original vs Log
    axes[i, 0].hist({OUTPUT_VAR}[col].dropna(), bins=30)
    axes[i, 0].set_title(f'Original Distribution: {col}')
    
    axes[i, 1].hist({OUTPUT_VAR}[f'{col}_log'].dropna(), bins=30)
    axes[i, 1].set_title(f'Log Distribution: {col}')

plt.tight_layout()
plt.show()
display({OUTPUT_VAR}.head())
""",

    # 应用巴特沃斯低通滤波器去除高频噪声，需插值处理缺失值。
    "filter_butterworth": """
# Butterworth Lowpass Filter for {VAR_NAME}
{OUTPUT_VAR} = {VAR_NAME}.select_dtypes(include=['number']).copy()

# Filter parameters
order = 4
fs = 1.0       # Sampling frequency (assume 1 Hz if unknown)
cutoff = 0.1   # Cutoff frequency (fraction of Nyquist)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

for col in {OUTPUT_VAR}.columns:
    # Handle NaNs before filtering
    data = {OUTPUT_VAR}[col].interpolate().fillna(method='bfill').fillna(method='ffill')
    {OUTPUT_VAR}[f'{col}_lowpass'] = butter_lowpass_filter(data.values, cutoff, fs, order)

# Visualize
plt.figure(figsize=(14, 6))
for col in {OUTPUT_VAR}.columns:
    if '_lowpass' not in col:
        plt.plot({OUTPUT_VAR}.index, {OUTPUT_VAR}[col], alpha=0.5, label=f'{col} (Original)')
        plt.plot({OUTPUT_VAR}.index, {OUTPUT_VAR}[f'{col}_lowpass'], linewidth=2, label=f'{col} (Filtered)')

plt.title(f'Butterworth Lowpass Filter (cutoff={cutoff})')
plt.legend()
plt.show()
""",

    # --- Exploratory Data Analysis (EDA) ---
    # 输出基础统计 describe、偏度与峰度、时间范围与采样间隔估计、缺失值数量与占比的汇总。
    "summary_stats": """
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

    # 绘制所有数值列的时序曲线；数据量过大时先下采样；启用中文、网格、图例并自适配布局。
    "line_plot": """
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

    # 对数值列插值后使用 Welch 方法计算功率谱密度并以半对数坐标绘制，用于识别周期成分。
    "spectral_analysis": """
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

    # 对最多三个数值列计算并绘制自相关函数（ACF），默认滞后阶数为 50。
    "autocorrelation": """
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

    # 对第一数值列进行 STL 分解，必要时推断频率并插值，展示趋势、季节项与残差。
    "decomposition": """
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

    # 将 DatetimeIndex 拆分为日期与小时，按均值汇聚形成透视表并绘制热力图观察日内/季节性模式。
    "heatmap_distribution": """
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
    # 计算滚动均值与标准差，标记超过 mean±3*std 的异常点，并在图中高亮显示。
    "threshold_sigma": """
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

    # 对数值列训练孤立森林（默认污染率 0.05），输出模型判定的异常点并可视化。
    "isolation_forest": """
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

    # 对第一数值列使用 ruptures 的二分段 L2 模型检测变点，返回并展示切分位置。
    "change_point": """
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
    # 对第一数值列按固定窗口计算移动平均作为趋势线，并与原序列对比绘制。
    "trend_ma": """
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

    # 对第一数值列计算指数加权移动平均作为趋势（span 可调），并与原序列对比绘制。
    "trend_ewma": """
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

    # 对第一数值列使用 LOWESS/LOESS 进行非参数平滑拟合，得到平滑趋势线。
    "trend_loess": """
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

    # 对第一数值列拟合二次多项式得到趋势线，并与原序列对比展示。
    "trend_polyfit": """
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

    # 对第一数值列进行 STL 分解并提取 trend 分量，绘制趋势随时间的变化。
    "trend_stl_trend": """
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

    # 为每个数值列单独绘制子图并垂直堆叠展示，便于逐列观察。
    "trend_basic_stacked": """
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

    # 将所有数值列叠加在同一坐标轴中展示，便于横向对比。
    "trend_basic_overlay": """
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

    # 将各数值列按网格布局分别绘制，自动计算网格大小以适配列数。
    "trend_basic_grid": """
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
    "load_csv": "加载 CSV 文件并显示前 5 行预览。支持自动识别列类型。",
    "import_variable": "从当前 Kernel 会话中引入已存在的 DataFrame 变量。",
    "smoothing_sg": "对数据框的数值列执行 Savitzky-Golay 平滑：先插值填补缺失值，再按指定窗口长度和多项式阶数进行滤波，结果以 _sg 后缀写入新列。",
    "smoothing_ma": "对数值列使用居中窗口计算移动平均，得到平滑序列并以 _ma 后缀写入新列。",
    "interpolation_time": "依据索引类型选择插值方式：DatetimeIndex 使用 time 插值，否则使用 linear 插值，并输出剩余缺失统计。",
    "interpolation_spline": "尝试三次样条插值以获取更平滑的曲线，不兼容时回退到线性插值。",
    "resampling_down": "当索引为 DatetimeIndex 时按指定规则进行重采样聚合：数值列取均值，非数值列取 first。",
    "alignment": "以基准数据框为时间轴，使用 merge_asof 按最近时间点对齐其他数据源（示例需替换 OTHER_DF）。",
    "feature_scaling": "对数值列进行 Z-Score 标准化和 Min-Max 归一化，生成对应后缀的新列。",
    "diff_transform": "计算一阶和二阶差分以消除趋势，并绘制差分后的时序对比图。",
    "outlier_clip": "将数值列中超过 1% 和 99% 分位数的极端值进行盖帽（截断）处理。",
    "feature_extraction_time": "从 DatetimeIndex 提取小时、星期、月份、是否周末等时间特征列。",
    "feature_lag": "生成指定滞后步数（1-3）的特征列，用于构建自回归模型数据集。",
    "transform_log": "对数值列进行 Log1p 变换以平滑分布，自动处理负值偏移，并绘制分布对比图。",
    "filter_butterworth": "应用巴特沃斯低通滤波器去除高频噪声，需插值处理缺失值。",
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
    "merge_dfs": "合并两个数据框，支持 inner, outer, left, right 连接方式。",
    "train_test_split": "将数据集分割为训练集和测试集。",
}

ALGORITHM_PORTS = {
    "merge_dfs": {
        "inputs": [{"name": "left", "type": "DataFrame"}, {"name": "right", "type": "DataFrame"}],
        "outputs": [{"name": "merged", "type": "DataFrame"}]
    },
    "train_test_split": {
        "inputs": [{"name": "data", "type": "DataFrame"}],
        "outputs": [
            {"name": "X_train", "type": "DataFrame"},
            {"name": "X_test", "type": "DataFrame"},
            {"name": "y_train", "type": "Series"},
            {"name": "y_test", "type": "Series"}
        ]
    },
    "trend_plot": {
        "inputs": [{"name": "df_in", "type": "DataFrame"}],
        "outputs": [{"name": "df_out", "type": "DataFrame"}]
    }
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
            
            # Determine ports based on category or specific ID
            inputs = []
            outputs = []
            
            if algo_id in ALGORITHM_PORTS:
                inputs = ALGORITHM_PORTS[algo_id].get("inputs", [])
                outputs = ALGORITHM_PORTS[algo_id].get("outputs", [])
            elif cat_label == "加载数据":
                inputs = []
                outputs = [{"name": "df_out", "type": "DataFrame"}]
            elif cat_label in ["数据预处理", "异常检测", "特征工程"]:
                inputs = [{"name": "df_in", "type": "DataFrame"}]
                outputs = [{"name": "df_out", "type": "DataFrame"}]
            elif cat_label in ["探索式分析", "趋势绘制"]:
                inputs = [{"name": "df_in", "type": "DataFrame"}]
                outputs = [] # Visualization nodes typically don't output a dataframe to next node in this simple model
                # But some EDA like summary_stats might. For now let's assume they are sinks or visualizers.
                if algo_id == "summary_stats":
                     outputs = [{"name": "df_out", "type": "DataFrame"}]
            
            # Determine nodeType (UI Component configuration)
            # Default to 'generic'
            node_type = "generic"
            
            # Configuration for custom nodes
            if algo_id == "load_csv":
                node_type = "csv_loader"
            elif algo_id == "plot_custom":
                node_type = "plot"
            elif algo_id == "import_variable":
                node_type = "generic" # We will handle variable-selector in GenericNode
            elif algo_id == "trend_plot":
                node_type = "trend"
            
            library[cat_label].append({
                "id": algo_id,
                "name": algo["name"],
                "description": description,
                "category": cat_label,
                "template": template,
                "inputs": inputs,
                "outputs": outputs,
                "nodeType": node_type,  # Frontend configuration
                "imports": ALGORITHM_IMPORTS.get(algo_id, []), # Add imports field
                # Keep backward compatibility fields if needed, or dummy them
                "docstring": template, 
                "signature": "",
                "module": "algorithm_templates",
                "args": ALGORITHM_PARAMETERS.get(algo_id, [])
            })
    
    # Inject custom 'Plot' node if not present in standard library
    # This corresponds to the frontend's PlotNode
    # Logic moved to standard registry
            
    return library
