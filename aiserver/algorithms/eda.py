from .base import Algorithm, AlgorithmParameter, Port

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

sampling_period = Algorithm(
    id="sampling_period",
    name="采样周期统计",
    category="eda",
    prompt="请对{VAR_NAME} 进行采样周期统计。计算每一列数据的实际采样周次。",
    parameters=[],
    imports=["import pandas as pd"],
    inputs=[Port(name="df_in")],
    outputs=[],  # No output node
    template="""
# Sampling Period Analysis for {VAR_NAME}
{OUTPUT_VAR} = {VAR_NAME}.copy()

if isinstance({OUTPUT_VAR}.index, pd.DatetimeIndex):
    # Initialize results list
    results = []
    
    # Iterate through each column
    for col in {OUTPUT_VAR}.columns:
        # Calculate time differences for non-null values in this column
        col_data = {OUTPUT_VAR}[col].dropna()
        
        if len(col_data) < 2:
            continue
            
        # Get the indices of non-null values
        non_null_indices = col_data.index
        
        # Calculate time differences between consecutive non-null points
        col_time_diffs = non_null_indices.to_series().diff().dropna()
        
        if not col_time_diffs.empty:
            # Calculate average sampling period in seconds
            avg_seconds = col_time_diffs.mean().total_seconds()
            if avg_seconds > 0:
                # Add to results list
                results.append({
                    "列名": col,
                    "平均采样周期": "平均采样周期",
                    "采样周期值": f"{avg_seconds:.0f}s"
                })
    
    # Create DataFrame from results
    if results:
        sampling_df = pd.DataFrame(results)
        display(sampling_df)
    else:
        print("没有足够的数据点来计算采样周期。")
else:
    print("错误: 数据索引不是时间索引，无法进行采样周期统计。")
"""
)

data_features = Algorithm(
    id="data_features",
    name="数据特征",
    category="eda",
    prompt="请对{VAR_NAME} 进行数据特征计算。使用pandas的describe()函数，计算各列的基本统计特征。",
    parameters=[],
    imports=["import pandas as pd"],
    inputs=[Port(name="df_in")],
    outputs=[],  # No output node
    template="""
# Data Features Analysis for {VAR_NAME}
{OUTPUT_VAR} = {VAR_NAME}.copy()

print("=== 数据基本统计特征 ===")
print()

# Calculate and display describe()
describe_result = {OUTPUT_VAR}.describe()
display(describe_result)

print()
print("=== 数据结构信息 ===")
print(f"数据形状: {OUTPUT_VAR}.shape")
print(f"列名: {list({OUTPUT_VAR}.columns)}")
print(f"数据类型:")
display({OUTPUT_VAR}.dtypes)

print()
print("=== 缺失值统计 ===")
missing_values = {OUTPUT_VAR}.isnull().sum()
missing_percentage = ({OUTPUT_VAR}.isnull().sum() / len({OUTPUT_VAR})) * 100
missing_stats = pd.DataFrame({"缺失值数量": missing_values, "缺失值百分比(%)": missing_percentage.round(2)})
display(missing_stats[missing_stats["缺失值数量"] > 0])

if missing_stats[missing_stats["缺失值数量"] > 0].empty:
    print("无缺失值")
"""
)

algorithms = [
    autocorrelation, decomposition, sampling_period, data_features
]

