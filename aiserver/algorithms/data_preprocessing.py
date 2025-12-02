from .base import Algorithm, AlgorithmParameter, Port

interpolation_spline = Algorithm(
    id="interpolation_spline",
    name="样条插值",
    category="data_preprocessing",
    prompt="请对{VAR_NAME} 进行样条插值 (Spline)。使用 pandas 的 interpolate(method='spline', order=3) 以获得更平滑的补全曲线。",
    parameters=[
        AlgorithmParameter(name="order", type="int", default=3, label="样条阶数", description="样条插值的阶数", min=1, max=5, priority="critical")
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
    name="降采样",
    category="data_preprocessing",
    prompt="请对{VAR_NAME} 进行降采样聚合。使用 pandas 的 resample() 将数据聚合到更低的时间分辨率（例如 '1min' 或 '1H'）；数值列使用 mean()，状态列使用 last() 或 max()。",
    parameters=[
        AlgorithmParameter(name="rule", type="str", default="1小时", label="频率规则", description="目标重采样频率", options=["15秒", "30秒", "1分钟", "5分钟", "15分钟", "30分钟", "1小时"], priority="critical"),
        AlgorithmParameter(name="agg_method", type="str", default="均值", label="聚合方法", options=["均值", "求和", "最小值", "最大值", "第一个值", "最后一个值", "中位数", "标准差", "方差", "计数"], description="降采样时使用的聚合函数", priority="critical")
    ],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    imports=["import pandas as pd"],
    template="""
# Downsampling (Aggregation) for {VAR_NAME}
# Requires DatetimeIndex
{OUTPUT_VAR} = {VAR_NAME}.copy()

if isinstance({OUTPUT_VAR}.index, pd.DatetimeIndex):
    # Map Chinese frequency to pandas frequency string
    freq_map = {
        "15秒": "15s",
        "30秒": "30s",
        "1分钟": "1T",
        "5分钟": "5T",
        "15分钟": "15T",
        "30分钟": "30T",
        "1小时": "1H"
    }
    
    # Map Chinese aggregation method to pandas function name
    agg_method_map = {
        "均值": "mean",
        "求和": "sum",
        "最小值": "min",
        "最大值": "max",
        "第一个值": "first",
        "最后一个值": "last",
        "中位数": "median",
        "标准差": "std",
        "方差": "var",
        "计数": "count"
    }
    
    # Get selected frequency in Chinese and map to pandas frequency
    chinese_rule = '{rule}'
    pandas_rule = freq_map.get(chinese_rule, "15s")  # Default to 1H if not found
    
    # Get selected aggregation method in Chinese and map to pandas function
    chinese_agg_method = '{agg_method}'
    agg_method = agg_method_map.get(chinese_agg_method, "mean")  # Default to mean if not found
    
    # Define aggregation dictionary: use selected method for all columns
    agg_dict = {col: agg_method for col in {OUTPUT_VAR}.columns}
            
    {OUTPUT_VAR} = {OUTPUT_VAR}.resample(pandas_rule).agg(agg_dict)
    print(f"Resampled to {chinese_rule} ({pandas_rule}) frequency. New shape: {{OUTPUT_VAR}.shape}")
    print(f"Aggregation method: {chinese_agg_method} ({agg_method})")
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
    prompt="请对 {VAR_NAME} 进行特征缩放。支持多种缩放方法，直接修改原始列。",
    parameters=[
        AlgorithmParameter(name="method", type="str", default="standard", label="缩放方法", options=["standard", "minmax", "robust", "maxabs"], description="选择缩放方法：standard（Z-score）、minmax（0-1归一化）、robust（鲁棒缩放）、maxabs（最大绝对值缩放）", priority="critical"),
        AlgorithmParameter(name="with_mean", type="bool", default=True, label="包含均值", description="对于standard和robust方法，是否减去均值", priority="non-critical"),
        AlgorithmParameter(name="with_std", type="bool", default=True, label="包含标准差", description="对于standard方法，是否除以标准差", priority="non-critical"),
        AlgorithmParameter(name="feature_range", type="str", default="(0, 1)", label="特征范围", description="对于minmax方法，指定目标范围，格式为'(min, max)'", priority="non-critical")
    ],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    imports=["import pandas as pd", "from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler"],
    template="""
# Feature Scaling for {VAR_NAME}
{OUTPUT_VAR} = {VAR_NAME}.copy()

# Select numeric columns only
numeric_cols = {OUTPUT_VAR}.select_dtypes(include=['number']).columns
if not numeric_cols.empty:
    # Get parameters
    method = '{method}'
    with_mean = {with_mean}
    with_std = {with_std}
    feature_range_str = '{feature_range}'
    
    # Parse feature range
    try:
        if feature_range_str.startswith('(') and feature_range_str.endswith(')'):
            feature_range = tuple(map(float, feature_range_str[1:-1].split(',')))
        else:
            feature_range = (0, 1)  # Default
    except:
        feature_range = (0, 1)  # Fallback
    
    # Apply selected scaler
    try:
        if method == 'standard':
            scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
        elif method == 'minmax':
            scaler = MinMaxScaler(feature_range=feature_range)
        elif method == 'robust':
            scaler = RobustScaler(with_centering=with_mean, with_scaling=with_std)
        elif method == 'maxabs':
            scaler = MaxAbsScaler()
        else:
            scaler = StandardScaler()  # Fallback
        
        # Scale in-place on original columns
        {OUTPUT_VAR}[numeric_cols] = scaler.fit_transform({OUTPUT_VAR}[numeric_cols])
        
        print(f"Applied {method} scaling to {len(numeric_cols)} columns")
        if method == 'standard':
            print(f"  with_mean: {with_mean}, with_std: {with_std}")
        elif method == 'minmax':
            print(f"  feature_range: {feature_range}")
        elif method == 'robust':
            print(f"  with_centering: {with_mean}, with_scaling: {with_std}")
        
        print(f"New shape: {{OUTPUT_VAR}.shape}")
        display({OUTPUT_VAR}.head())
    except Exception as e:
        print(f"Scaling failed: {e}")
else:
    print("No numeric columns found for scaling")
"""
)

diff_transform = Algorithm(
    id="diff_transform",
    name="差分变换",
    category="data_preprocessing",
    prompt="请对 {VAR_NAME} 进行差分变换，以消除趋势并使数据平稳。可配置差分阶数和滞后步数。",
    parameters=[
        AlgorithmParameter(name="order", type="int", default=1, label="差分阶数", description="差分的阶数，1为一阶差分，2为二阶差分等", min=1, max=5, step=1, priority="critical"),
        AlgorithmParameter(name="periods", type="int", default=1, label="滞后步数", description="差分的滞后步数，默认1", min=1, max=10, step=1, priority="critical"),
        AlgorithmParameter(name="axis", type="int", default=0, label="差分轴", options=[0, 1], description="沿哪个轴进行差分，0=行（时间轴），1=列", min=0, max=1, step=1, priority="non-critical"),
        AlgorithmParameter(name="fill_method", type="str", default="", label="填充方法", options=["", "ffill", "bfill"], description="差分后缺失值的填充方法，留空则不填充", priority="non-critical")
    ],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    imports=["import pandas as pd"],
    template="""
# Difference Transform for {VAR_NAME}
{OUTPUT_VAR} = {VAR_NAME}.select_dtypes(include=['number']).copy()

# Apply difference transform
order = {order}
periods = {periods}
axis = {axis}
fill_method = '{fill_method}'

# Perform difference transform
try:
    # Apply difference multiple times for higher orders
    for i in range(order):
        {OUTPUT_VAR} = {OUTPUT_VAR}.diff(periods=periods, axis=axis)
    
    # Fill missing values if specified
    if fill_method:
        {OUTPUT_VAR} = {OUTPUT_VAR}.fillna(method=fill_method)
    
    print(f"Applied {order}nd order difference with periods={periods} along axis={axis}")
    if fill_method:
        print(f"Filled missing values using {fill_method}")
    print(f"New shape: {{OUTPUT_VAR}.shape}")
    display({OUTPUT_VAR}.head())
except Exception as e:
    print(f"Difference transform failed: {e}")
"""
)

data_fill = Algorithm(
    id="data_fill",
    name="数据填充",
    category="data_preprocessing",
    prompt="请对 {VAR_NAME} 进行缺失值填充。支持多种填充方法，包括均值、中位数、众数、前向填充、后向填充、常数填充等。",
    parameters=[
        AlgorithmParameter(name="method", type="str", default="均值", label="填充方法", options=["均值", "中位数", "众数", "前向填充", "后向填充", "常数", "线性插值", "最近邻插值"], description="选择缺失值填充方法", priority="critical"),
        AlgorithmParameter(name="value", type="float", default=0.0, label="填充值", description="当使用常数填充时，指定填充的值", priority="non-critical"),
        AlgorithmParameter(name="axis", type="int", default=0, label="填充轴", options=[0, 1], description="沿哪个轴进行填充，0=按列填充，1=按行填充", priority="non-critical"),
        AlgorithmParameter(name="limit", type="int", default=0, label="填充限制", description="限制连续缺失值的填充数量，0表示无限制", min=0, max=1000, priority="non-critical")
    ],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    imports=["import pandas as pd"],
    template="""
# Data Fill for {VAR_NAME}
{OUTPUT_VAR} = {VAR_NAME}.copy()

print("=== 数据填充前 ===")
print(f"数据形状: {{OUTPUT_VAR}.shape}")
print(f"缺失值总数: {{OUTPUT_VAR}.isnull().sum().sum()}")
print("各列缺失值数量:")
display({OUTPUT_VAR}.isnull().sum()[{OUTPUT_VAR}.isnull().sum() > 0])
print()

# Get fill parameters
method = '{method}'
value = {value}
axis = {axis}
limit = {limit} if {limit} > 0 else None

# Perform filling
print(f"使用方法 '{method}' 进行数据填充...")

# Map Chinese method name to pandas method
method_map = {
    "均值": "mean",
    "中位数": "median",
    "众数": "mode",
    "前向填充": "ffill",
    "后向填充": "bfill",
    "常数": "constant",
    "线性插值": "linear",
    "最近邻插值": "nearest"
}

pandas_method = method_map.get(method, "mean")

# Apply filling based on method
filled_cols = []
for col in {OUTPUT_VAR}.columns:
    if {OUTPUT_VAR}[col].isnull().sum() > 0:
        # Skip non-numeric columns for mean/median/mode
        if pandas_method in ["mean", "median"] and not pd.api.types.is_numeric_dtype({OUTPUT_VAR}[col]):
            print(f"跳过非数值列 '{col}'，使用前向填充")
            {OUTPUT_VAR}[col] = {OUTPUT_VAR}[col].fillna(method="ffill", limit=limit, axis=axis)
        elif pandas_method == "mode" and not pd.api.types.is_numeric_dtype({OUTPUT_VAR}[col]):
            print(f"跳过非数值列 '{col}'，使用前向填充")
            {OUTPUT_VAR}[col] = {OUTPUT_VAR}[col].fillna(method="ffill", limit=limit, axis=axis)
        elif pandas_method == "constant":
            {OUTPUT_VAR}[col] = {OUTPUT_VAR}[col].fillna(value=value, limit=limit)
        elif pandas_method in ["ffill", "bfill"]:
            {OUTPUT_VAR}[col] = {OUTPUT_VAR}[col].fillna(method=pandas_method, limit=limit, axis=axis)
        else:  # Interpolation methods
            try:
                {OUTPUT_VAR}[col] = {OUTPUT_VAR}[col].interpolate(method=pandas_method, limit=limit, axis=axis)
            except Exception as e:
                print(f"插值填充失败，使用前向填充: {e}")
                {OUTPUT_VAR}[col] = {OUTPUT_VAR}[col].fillna(method="ffill", limit=limit, axis=axis)
        filled_cols.append(col)

print(f"填充了 {len(filled_cols)} 列")
print()

print("=== 数据填充后 ===")
print(f"数据形状: {{OUTPUT_VAR}.shape}")
print(f"缺失值总数: {{OUTPUT_VAR}.isnull().sum().sum()}")
if {OUTPUT_VAR}.isnull().sum().sum() > 0:
    print("剩余缺失值数量:")
    display({OUTPUT_VAR}.isnull().sum()[{OUTPUT_VAR}.isnull().sum() > 0])
else:
    print("所有缺失值已填充完成")

print()
display({OUTPUT_VAR}.head())
"""
)

algorithms = [
    interpolation_spline, resampling_down, alignment, feature_scaling, diff_transform, data_fill
]