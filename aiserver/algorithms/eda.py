from .base import Algorithm, AlgorithmParameter, Port

plot_custom = Algorithm(
    id="plot_custom",
    name="通用绘图 (Plot)",
    category="eda",
    prompt="请对 {VAR_NAME} 进行绘图。根据用户选择的图表类型（折线图、柱状图、散点图）和指定列进行可视化。",
    parameters=[
        AlgorithmParameter(name="plot_type", type="str", default="line", label="图表类型", options=["line", "bar", "scatter"], description="图表的类型"),
        AlgorithmParameter(name="column", type="str", default="", label="列名", description="要绘制的列 (Y轴)")
    ],
    imports=["import matplotlib.pyplot as plt"],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    template="""
# Generic Plot
# Plot type: {plot_type}, Column: {column}
if '{column}':
    {VAR_NAME}.plot(kind='{plot_type}', y='{column}')
else:
    {VAR_NAME}.plot(kind='{plot_type}')
plt.show()
"""
)

summary_stats = Algorithm(
    id="summary_stats",
    name="时序统计摘要",
    category="eda",
    prompt="请对{VAR_NAME} 执行时序统计摘要分析。\\n代码应包含以下步骤:\\n1. 基础统计量：使用 pandas 的 describe() 方法计算均值、标准差、最小值、最大值及四分位数。\\n2. 分布特征：计算数据的偏度 (skewness) 和峰度 (kurtosis) 以评估数据分布形态。\\n3. 时间特性分析：\\n   - 计算时间覆盖范围（起始时间、结束时间、总时长）。\\n   - 推断或计算平均采样间隔/频率。\\n4. 数据质量评估：统计缺失值的数量及占比。\\n请将所有结果汇总为一个清晰的 DataFrame 或字典格式返回，以便于查看。",
    parameters=[],
    imports=["import pandas as pd"],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    template="""
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
"""
)

line_plot = Algorithm(
    id="line_plot",
    name="多尺度时序曲线",
    category="eda",
    prompt="请对{VAR_NAME} 绘制时序曲线图。使用 matplotlib 或 seaborn，并满足：1. X轴时间格式化；2. 支持缩放；3. 添加网格、图例与中文标签；4. 数据量过大时先降采样再绘图。",
    parameters=[],
    imports=["import matplotlib.pyplot as plt", "import seaborn as sns", "import pandas as pd"],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    template="""
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
"""
)

spectral_analysis = Algorithm(
    id="spectral_analysis",
    name="频谱分析 (PSD)",
    category="eda",
    prompt="请对{VAR_NAME} 进行频域分析，计算并绘制功率谱密度 (PSD)。使用 scipy.signal.welch，并以对数坐标显示频率和功率，用于识别周期性成分。",
    parameters=[],
    imports=["import numpy as np", "import matplotlib.pyplot as plt", "from scipy.signal import welch"],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    template="""
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
"""
)

autocorrelation = Algorithm(
    id="autocorrelation",
    name="自相关分析 (ACF)",
    category="eda",
    prompt="请对{VAR_NAME} 进行自相关分析。计算并绘制 ACF 图，使用 statsmodels.graphics.tsaplots.plot_acf，以发现周期性模式。",
    parameters=[
        AlgorithmParameter(name="lags", type="int", default=50, label="滞后数", description="绘制的滞后数量", min=10, max=200)
    ],
    imports=["import matplotlib.pyplot as plt", "from statsmodels.graphics.tsaplots import plot_acf"],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    template="""
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
"""
)

decomposition = Algorithm(
    id="decomposition",
    name="STL 分解",
    category="eda",
    prompt="请对{VAR_NAME} 执行 STL 分解 (Seasonal-Trend decomposition using LOESS)。将数据分解为趋势、季节与残差，并绘制分解结果图。",
    parameters=[],
    imports=["import matplotlib.pyplot as plt", "from statsmodels.tsa.seasonal import STL", "import pandas as pd"],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    template="""
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
"""
)

heatmap_distribution = Algorithm(
    id="heatmap_distribution",
    name="时序热力图",
    category="eda",
    prompt="请基于 {VAR_NAME} 绘制时序热力图，用于观察分布模式（例如 X=日期，Y=小时，颜色=数值）。用于发现日内模式或季节性变化。",
    parameters=[],
    imports=["import seaborn as sns", "import matplotlib.pyplot as plt", "import pandas as pd"],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    template="""
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
"""
)

algorithms = [
    plot_custom, summary_stats, line_plot, spectral_analysis,
    autocorrelation, decomposition, heatmap_distribution
]

