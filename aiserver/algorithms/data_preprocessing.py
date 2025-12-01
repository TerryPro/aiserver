from .base import Algorithm, AlgorithmParameter, Port

smoothing_sg = Algorithm(
    id="smoothing_sg",
    name="Savitzky-Golay 平滑",
    category="data_preprocessing",
    prompt="请对{VAR_NAME} 执行 Savitzky-Golay 平滑处理。使用 scipy.signal.savgol_filter 去除高频噪声并保留波形特征，并根据数据特性选择或建议合适的窗口长度与多项式阶数。",
    parameters=[
        AlgorithmParameter(name="window_length", type="int", default=11, label="窗口长度", description="必须是奇数且大于多项式阶数", min=3, step=2),
        AlgorithmParameter(name="polyorder", type="int", default=3, label="多项式阶数", description="用于拟合样本的多项式阶数", min=1, max=5)
    ],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    imports=["import pandas as pd", "import numpy as np", "from scipy.signal import savgol_filter"],
    template="""
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
"""
)

smoothing_ma = Algorithm(
    id="smoothing_ma",
    name="移动平均平滑",
    category="data_preprocessing",
    prompt="请对{VAR_NAME} 执行移动平均平滑。使用 pandas 的 rolling().mean() 方法，并根据采样频率选择合理的窗口大小。",
    parameters=[
        AlgorithmParameter(name="window_size", type="int", default=5, label="窗口大小", description="移动窗口的大小", min=1)
    ],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    imports=["import pandas as pd", "import numpy as np"],
    template="""
# Moving Average Smoothing for {VAR_NAME}
{OUTPUT_VAR} = {VAR_NAME}.copy()
numeric_cols = {OUTPUT_VAR}.select_dtypes(include=[np.number]).columns
window_size = {window_size}  # Adjust window size as needed

for col in numeric_cols:
    {OUTPUT_VAR}[f'{col}_ma'] = {OUTPUT_VAR}[col].rolling(window=window_size, center=True).mean()

# Display first few rows
{OUTPUT_VAR}.head()
"""
)

interpolation_time = Algorithm(
    id="interpolation_time",
    name="时间加权插值",
    category="data_preprocessing",
    prompt="请对{VAR_NAME} 的缺失值进行时间加权插值。不等间隔数据使用 pandas 的 interpolate(method='time') 以确保物理意义的准确性。",
    parameters=[],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    imports=["import pandas as pd"],
    template="""
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
"""
)

interpolation_spline = Algorithm(
    id="interpolation_spline",
    name="样条插值",
    category="data_preprocessing",
    prompt="请对{VAR_NAME} 进行样条插值 (Spline)。使用 pandas 的 interpolate(method='spline', order=3) 以获得更平滑的补全曲线。",
    parameters=[
        AlgorithmParameter(name="order", type="int", default=3, label="样条阶数", description="样条插值的阶数", min=1, max=5)
    ],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    imports=["import pandas as pd", "import numpy as np"],
    template="""
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
"""
)

resampling_down = Algorithm(
    id="resampling_down",
    name="降采样 (聚合)",
    category="data_preprocessing",
    prompt="请对{VAR_NAME} 进行降采样聚合。使用 pandas 的 resample() 将数据聚合到更低的时间分辨率（例如 '1min' 或 '1H'）；数值列使用 mean()，状态列使用 last() 或 max()。",
    parameters=[
        AlgorithmParameter(name="rule", type="str", default="1H", label="频率规则", description="目标频率 (例如 '1H', '1D', '15T')", options=["1T", "5T", "15T", "30T", "1H", "6H", "12H", "1D", "1W", "1M"])
    ],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    imports=["import pandas as pd"],
    template="""
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
    print(f"Resampled to {rule} frequency. New shape: {{OUTPUT_VAR}.shape}")
    display({OUTPUT_VAR}.head())
else:
    print("Error: {VAR_NAME} index is not a DatetimeIndex. Cannot resample.")
"""
)

alignment = Algorithm(
    id="alignment",
    name="多源数据对齐",
    category="data_preprocessing",
    prompt="请以 {VAR_NAME} 为基准执行多源时间对齐。使用 pandas 的 merge_asof 方法，将其他数据对齐到该时间轴。",
    parameters=[],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    imports=["import pandas as pd"],
    template="""
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
"""
)

feature_scaling = Algorithm(
    id="feature_scaling",
    name="数据标准化/归一化",
    category="data_preprocessing",
    prompt="请对 {VAR_NAME} 进行特征缩放。提供 Z-Score 标准化和 Min-Max 归一化两种结果，便于后续模型训练。",
    parameters=[],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    imports=["import pandas as pd", "from sklearn.preprocessing import StandardScaler", "MinMaxScaler"],
    template="""
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
"""
)

diff_transform = Algorithm(
    id="diff_transform",
    name="差分变换",
    category="data_preprocessing",
    prompt="请对 {VAR_NAME} 进行一阶和二阶差分处理，以消除趋势并使数据平稳，绘制差分后的时序图。",
    parameters=[],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    imports=["import pandas as pd", "import matplotlib.pyplot as plt"],
    template="""
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
"""
)

outlier_clip = Algorithm(
    id="outlier_clip",
    name="离群值盖帽 (Winsorization)",
    category="data_preprocessing",
    prompt="请对 {VAR_NAME} 进行离群值盖帽处理。将超出 1% 和 99% 分位数的值限制在边界范围内，以减少极端值的影响。",
    parameters=[
        AlgorithmParameter(name="lower_quantile", type="float", default=0.01, label="下分位数", description="裁剪下限 (0.0-1.0)", min=0.0, max=0.5, step=0.01),
        AlgorithmParameter(name="upper_quantile", type="float", default=0.99, label="上分位数", description="裁剪上限 (0.0-1.0)", min=0.5, max=1.0, step=0.01)
    ],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    imports=["import pandas as pd"],
    template="""
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
"""
)

feature_extraction_time = Algorithm(
    id="feature_extraction_time",
    name="时间特征提取",
    category="data_preprocessing",
    prompt="请从 {VAR_NAME} 的时间索引中提取特征。生成'小时'、'星期几'、'月份'、'是否周末'等新列，用于机器学习模型输入。",
    parameters=[],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    imports=["import pandas as pd"],
    template="""
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
"""
)

feature_lag = Algorithm(
    id="feature_lag",
    name="滞后特征生成",
    category="data_preprocessing",
    prompt="请对 {VAR_NAME} 生成滞后特征。创建滞后 1 至 3 个时间步的列（lag_1, lag_2, lag_3），用于自回归分析。",
    parameters=[
        AlgorithmParameter(name="max_lag", type="int", default=3, label="最大滞后阶数", description="生成的最大滞后阶数", min=1, max=24)
    ],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    imports=["import pandas as pd"],
    template="""
# Lag Feature Generation for {VAR_NAME}
{OUTPUT_VAR} = {VAR_NAME}.select_dtypes(include=['number']).copy()
target_cols = {OUTPUT_VAR}.columns
lags = [1, 2, 3]

for col in target_cols:
    for lag in lags:
        {OUTPUT_VAR}[f'{col}_lag_{lag}'] = {OUTPUT_VAR}[col].shift(lag)

# Drop rows with NaNs created by shifting
{OUTPUT_VAR}.dropna(inplace=True)

print(f"Generated lag features for {lags}. New shape: {{OUTPUT_VAR}.shape}")
display({OUTPUT_VAR}.head())
"""
)

transform_log = Algorithm(
    id="transform_log",
    name="对数变换",
    category="data_preprocessing",
    prompt="请对 {VAR_NAME} 进行对数变换。使用 log1p 处理以稳定方差，并绘制变换前后的分布对比图。",
    parameters=[],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    imports=["import pandas as pd", "import numpy as np", "import matplotlib.pyplot as plt"],
    template="""
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
"""
)

filter_butterworth = Algorithm(
    id="filter_butterworth",
    name="巴特沃斯低通滤波",
    category="data_preprocessing",
    prompt="请对 {VAR_NAME} 应用巴特沃斯低通滤波器。设置截止频率和阶数，去除高频噪声，保留主要趋势信号。",
    parameters=[],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    imports=["import pandas as pd", "import numpy as np", "from scipy.signal import butter, filtfilt", "import matplotlib.pyplot as plt"],
    template="""
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
"""
)

merge_dfs = Algorithm(
    id="merge_dfs",
    name="数据合并 (Merge)",
    category="data_preprocessing",
    prompt="请合并两个数据框 {left} 和 {right}。根据指定的合并方式（inner, outer, left, right）和连接键进行 pd.merge 操作。",
    parameters=[
        AlgorithmParameter(name="how", type="str", default="inner", label="合并方式", options=["inner", "outer", "left", "right"], description="执行合并的方式"),
        AlgorithmParameter(name="on", type="str", default="", label="合并列", description="用于连接的列名或索引级别名。留空则使用索引。", widget="column-selector")
    ],
    inputs=[Port(name="left"), Port(name="right")],
    outputs=[Port(name="merged")],
    imports=["import pandas as pd"],
    template="""
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
            
        print(f"Merged shape: {{merged}.shape}")
        display({merged}.head())
except Exception as e:
    print(f"Merge failed: {e}")
"""
)

train_test_split_algo = Algorithm(
    id="train_test_split",
    name="训练/测试集分割",
    category="data_preprocessing",
    prompt="请将 {data} 分割为训练集和测试集。使用 sklearn.model_selection.train_test_split，返回 X_train, X_test, y_train, y_test。",
    parameters=[
        AlgorithmParameter(name="test_size", type="float", default=0.2, label="测试集比例", description="包含在测试拆分中的数据集比例", min=0.01, max=0.99, step=0.05),
        AlgorithmParameter(name="target_column", type="str", default="target", label="目标列", description="目标变量列名 (y)", widget="column-selector"),
        AlgorithmParameter(name="random_state", type="int", default=42, label="随机种子", description="控制拆分前的数据打乱")
    ],
    inputs=[Port(name="data")],
    outputs=[Port(name="X_train"), Port(name="X_test"), Port(name="y_train"), Port(name="y_test")],
    imports=["from sklearn.model_selection import train_test_split", "import pandas as pd"],
    template="""
# Train/Test Split
# Input: {data}
# Outputs: {X_train}, {X_test}, {y_train}, {y_test}

try:
    target = '{target_column}'
    if target not in {data}.columns:
        print(f"Error: Target column '{target}' not found in DataFrame.")
    else:
        X = {data}.drop(columns=[target])
        y = {data}[target]
        
        {X_train}, {X_test}, {y_train}, {y_test} = train_test_split(
            X, y, test_size={test_size}, random_state={random_state}
        )
        
        print(f"Train shape: X={{X_train}.shape}, y={{y_train}.shape}")
        print(f"Test shape:  X={{X_test}.shape},  y={{y_test}.shape}")
except Exception as e:
    print(f"Split failed: {e}")
"""
)

algorithms = [
    smoothing_sg, smoothing_ma, interpolation_time, interpolation_spline, resampling_down,
    alignment, feature_scaling, diff_transform, outlier_clip, feature_extraction_time,
    feature_lag, transform_log, filter_butterworth, merge_dfs, train_test_split_algo
]




