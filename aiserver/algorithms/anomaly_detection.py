from .base import Algorithm, AlgorithmParameter, Port

threshold_sigma = Algorithm(
    id="threshold_sigma",
    name="3-Sigma 阈值检测",
    category="anomaly_detection",
    prompt="请在 {VAR_NAME} 上应用 3-Sigma 异常检测。计算移动窗口均值与标准差，将超过 mean ± n*std 的点标记为异常，并在原图用红色标记异常点。",
    parameters=[
        AlgorithmParameter(name="columns", type="list", default=[], label="检测列", description="要检测异常值的列", widget="column-selector", priority="critical"),
        AlgorithmParameter(name="window", type="int", default=20, label="窗口大小", description="移动窗口大小", min=5, max=300, step=5, priority="critical"),
        AlgorithmParameter(name="sigma", type="float", default=3.0, label="标准差倍数", description="标准差倍数，用于计算异常值边界", min=1.0, max=5.0, step=0.5, priority="critical")
    ],
    imports=["import numpy as np", "import pandas as pd", "import matplotlib.pyplot as plt"],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    template="""
# 3-Sigma Anomaly Detection for {VAR_NAME}
df_anom = {VAR_NAME}.copy()

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体显示中文
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

columns = {columns}
window = {window}
sigma = {sigma}

if not columns:
    columns = df_anom.select_dtypes(include=['number']).columns.tolist()

# 存储所有异常值数量
total_anomalies = 0

# 为每个列执行3-Sigma异常检测
for col in columns:
    data = df_anom[col]
    
    # 计算移动窗口均值和标准差
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    
    # 计算上下界
    upper_bound = rolling_mean + sigma * rolling_std
    lower_bound = rolling_mean - sigma * rolling_std
    
    # 标记异常值
    df_anom[f'{col}_rolling_mean'] = rolling_mean
    df_anom[f'{col}_rolling_std'] = rolling_std
    df_anom[f'{col}_upper_bound'] = upper_bound
    df_anom[f'{col}_lower_bound'] = lower_bound
    df_anom[f'{col}_is_anomaly'] = (data > upper_bound) | (data < lower_bound)
    
    # 统计异常值数量
    anomalies_count = df_anom[f'{col}_is_anomaly'].sum()
    total_anomalies += anomalies_count
    
    # 获取异常值数据
    anomalies = data[df_anom[f'{col}_is_anomaly']]
    
    # 可视化结果
    plt.figure(figsize=(15, 8))
    
    # 原始数据与异常值
    plt.subplot(2, 1, 1)
    plt.plot(data.index, data, label='原始数据', alpha=0.6)
    plt.plot(rolling_mean.index, rolling_mean, 'k--', label=f'移动平均线 (窗口={window})', alpha=0.8)
    plt.fill_between(rolling_mean.index, lower_bound, upper_bound, color='gray', alpha=0.2, label=f'{sigma}-Sigma 范围')
    plt.scatter(anomalies.index, anomalies, color='red', label='异常值', s=50, alpha=0.8, zorder=5)
    plt.title(f'{sigma}-Sigma 异常检测: {col}')
    plt.xlabel('时间' if isinstance(data.index, pd.DatetimeIndex) else '索引')
    plt.ylabel(col)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 异常值分布
    plt.subplot(2, 1, 2)
    plt.plot(data.index, df_anom[f'{col}_is_anomaly'].astype(int), label='异常值标记', color='red')
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.title(f'{sigma}-Sigma 异常检测: {col} 异常值分布')
    plt.xlabel('时间' if isinstance(data.index, pd.DatetimeIndex) else '索引')
    plt.ylabel('是否异常')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"列 '{col}' 发现 {anomalies_count} 个异常值。")

print(f"\n总计发现 {total_anomalies} 个异常值。")

# 将结果输出到OUTPUT_VAR
{OUTPUT_VAR} = df_anom
"""
)

isolation_forest = Algorithm(
    id="isolation_forest",
    name="孤立森林检测",
    category="anomaly_detection",
    prompt="请对 {VAR_NAME} 执行孤立森林 (Isolation Forest) 异常检测。设置污染率 (contamination) 参数，并在原图用红色标记识别出的异常点。",
    parameters=[
        AlgorithmParameter(name="contamination", type="float", default=0.05, label="污染率", description="预期的异常值比例", min=0.001, max=0.5, step=0.01)
    ],
    imports=["from sklearn.ensemble import IsolationForest", "import matplotlib.pyplot as plt", "import pandas as pd"],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    template="""
# Isolation Forest Anomaly Detection for {VAR_NAME}
df_iso = {VAR_NAME}.select_dtypes(include=['number']).dropna()

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体显示中文
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

model = IsolationForest(contamination=0.05, random_state=42)
df_iso['anomaly'] = model.fit_predict(df_iso)

# -1 indicates anomaly, 1 indicates normal
anomalies = df_iso[df_iso['anomaly'] == -1]

target_col = df_iso.columns[0] # Visualize first column

plt.figure(figsize=(15, 6))
plt.plot(df_iso.index, df_iso[target_col], label='正常数据', color='blue', alpha=0.6)
plt.scatter(anomalies.index, anomalies[target_col], color='red', label='异常值', s=20, zorder=5)
plt.title(f'孤立森林异常检测: {target_col}')
plt.legend()
plt.show()
"""
)

change_point = Algorithm(
    id="change_point",
    name="变点检测",
    category="anomaly_detection",
    prompt="请在 {VAR_NAME} 中检测统计特性发生突变的时间点 (Change Point Detection)。可以使用 ruptures 库或基于滑动窗口的统计差异检测方法。",
    parameters=[],
    imports=["import matplotlib.pyplot as plt", "import ruptures as rpt", "import numpy as np"],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    template="""
# Change Point Detection for {VAR_NAME}
# Uses binary segmentation with L2 cost
df_cp = {VAR_NAME}.select_dtypes(include=['number']).dropna()
target_col = df_cp.columns[0]
signal = df_cp[target_col].values

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体显示中文
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

# Detection
algo = rpt.Binseg(model="l2").fit(signal)
result = algo.predict(n_bkps=5) # Detect 5 change points

# Display
rpt.display(signal, result, figsize=(15, 6))
plt.title(f'变点检测: {target_col}')
plt.show()
"""
)
# 4. Z-score 异常检测算法
zscore_anomaly = Algorithm(
    id="anomaly_zscore",
    name="Z-score 异常检测",
    category="anomaly_detection",
    prompt="请使用 Z-score 方法检测 {VAR_NAME} 中的异常值。计算每个数据点的 Z-score，将绝对值大于阈值的数据点标记为异常，并绘制原始数据与异常值标记图。",
    parameters=[
        AlgorithmParameter(name="columns", type="list", default=[], label="检测列", description="要检测异常值的列", widget="column-selector", priority="critical"),
        AlgorithmParameter(name="threshold", type="float", default=3.0, label="异常阈值", description="Z-score 绝对值大于该值的数据点被标记为异常", min=1.0, max=5.0, step=0.5, priority="critical")
    ],
    imports=["import numpy as np", "import pandas as pd", "import matplotlib.pyplot as plt"],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    template="""# Z-score Anomaly Detection for {VAR_NAME}
{OUTPUT_VAR} = {VAR_NAME}.copy()

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体显示中文
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

columns = {columns}
threshold = {threshold}

if not columns:
    columns = {OUTPUT_VAR}.select_dtypes(include=['number']).columns.tolist()

for col in columns:
    # 计算全局Z-score
    mean = {OUTPUT_VAR}[col].mean()
    std = {OUTPUT_VAR}[col].std()
    {OUTPUT_VAR}[f'{col}_zscore'] = ({OUTPUT_VAR}[col] - mean) / std
    
    # 标记异常值
    {OUTPUT_VAR}[f'{col}_anomaly'] = np.abs({OUTPUT_VAR}[f'{col}_zscore']) > threshold

# 可视化结果
for col in columns:
    plt.figure(figsize=(15, 6))
    plt.plot({OUTPUT_VAR}.index, {OUTPUT_VAR}[col], label='原始数据')
    anomalies = {OUTPUT_VAR}[{OUTPUT_VAR}[f'{col}_anomaly']]
    plt.scatter(anomalies.index, anomalies[col], color='red', label='异常值', s=50, alpha=0.8)
    plt.title(f'Z-score 异常检测: {col}')
    plt.xlabel('时间' if isinstance({OUTPUT_VAR}.index, pd.DatetimeIndex) else '索引')
    plt.ylabel(col)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
"""
)

# 5. IQR（四分位距）异常检测算法
iqr_anomaly = Algorithm(
    id="anomaly_iqr",
    name="IQR 异常检测",
    category="anomaly_detection",
    prompt="请使用 IQR（四分位距）方法检测 {VAR_NAME} 中的异常值。计算每个数据点的 IQR，将超出上下界的数据点标记为异常，并绘制原始数据与异常值标记图。",
    parameters=[
        AlgorithmParameter(name="columns", type="list", default=[], label="检测列", description="要检测异常值的列", widget="column-selector", priority="critical"),
        AlgorithmParameter(name="multiplier", type="float", default=1.5, label="IQR 倍数", description="IQR 上下界的倍数，通常为1.5或3.0", min=1.0, max=5.0, step=0.5, priority="critical")
    ],
    imports=["import numpy as np", "import pandas as pd", "import matplotlib.pyplot as plt"],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    template="""# IQR Anomaly Detection for {VAR_NAME}
{OUTPUT_VAR} = {VAR_NAME}.copy()

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体显示中文
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

columns = {columns}
multiplier = {multiplier}

if not columns:
    columns = {OUTPUT_VAR}.select_dtypes(include=['number']).columns.tolist()

for col in columns:
    # 计算IQR
    q1 = {OUTPUT_VAR}[col].quantile(0.25)
    q3 = {OUTPUT_VAR}[col].quantile(0.75)
    iqr = q3 - q1
    
    # 计算上下界
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    # 标记异常值
    {OUTPUT_VAR}[f'{col}_lower_bound'] = lower_bound
    {OUTPUT_VAR}[f'{col}_upper_bound'] = upper_bound
    {OUTPUT_VAR}[f'{col}_anomaly'] = ({OUTPUT_VAR}[col] < lower_bound) | ({OUTPUT_VAR}[col] > upper_bound)

# 可视化结果
for col in columns:
    plt.figure(figsize=(15, 6))
    plt.plot({OUTPUT_VAR}.index, {OUTPUT_VAR}[col], label='原始数据')
    anomalies = {OUTPUT_VAR}[{OUTPUT_VAR}[f'{col}_anomaly']]
    plt.scatter(anomalies.index, anomalies[col], color='red', label='异常值', s=50, alpha=0.8)
    plt.axhline(y={OUTPUT_VAR}[f'{col}_lower_bound'].iloc[0], color='green', linestyle='--', label=f'下界 ({multiplier}x IQR)')
    plt.axhline(y={OUTPUT_VAR}[f'{col}_upper_bound'].iloc[0], color='green', linestyle='--', label=f'上界 ({multiplier}x IQR)')
    plt.title(f'IQR 异常检测: {col}')
    plt.xlabel('时间' if isinstance({OUTPUT_VAR}.index, pd.DatetimeIndex) else '索引')
    plt.ylabel(col)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
"""
)

# 6. 移动窗口 Z-score 异常检测算法
moving_window_zscore_anomaly = Algorithm(
    id="anomaly_moving_window_zscore",
    name="移动窗口 Z-score 异常检测",
    category="anomaly_detection",
    prompt="请使用移动窗口 Z-score 方法检测 {VAR_NAME} 中的异常值。在滑动窗口内计算 Z-score，将绝对值大于阈值的数据点标记为异常，并绘制原始数据与异常值标记图。",
    parameters=[
        AlgorithmParameter(name="columns", type="list", default=[], label="检测列", description="要检测异常值的列", widget="column-selector", priority="critical"),
        AlgorithmParameter(name="window", type="int", default=60, label="窗口大小", description="计算Z-score的滑动窗口大小", min=10, max=300, step=10, priority="critical"),
        AlgorithmParameter(name="threshold", type="float", default=3.0, label="异常阈值", description="Z-score 绝对值大于该值的数据点被标记为异常", min=1.0, max=5.0, step=0.5, priority="critical"),
        AlgorithmParameter(name="center", type="bool", default=True, label="居中对齐", description="是否在窗口中居中对齐Z-score", priority="non-critical")
    ],
    imports=["import numpy as np", "import pandas as pd", "import matplotlib.pyplot as plt"],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    template="""# Moving Window Z-score Anomaly Detection for {VAR_NAME}
{OUTPUT_VAR} = {VAR_NAME}.copy()

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体显示中文
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

columns = {columns}
window = {window}
threshold = {threshold}
center = {center}

if not columns:
    columns = {OUTPUT_VAR}.select_dtypes(include=['number']).columns.tolist()

for col in columns:
    # 计算移动窗口均值和标准差
    rolling_mean = {OUTPUT_VAR}[col].rolling(window=window, center=center).mean()
    rolling_std = {OUTPUT_VAR}[col].rolling(window=window, center=center).std()
    
    # 计算移动窗口Z-score
    {OUTPUT_VAR}[f'{col}_rolling_mean'] = rolling_mean
    {OUTPUT_VAR}[f'{col}_rolling_std'] = rolling_std
    {OUTPUT_VAR}[f'{col}_zscore'] = ({OUTPUT_VAR}[col] - rolling_mean) / rolling_std
    
    # 标记异常值
    {OUTPUT_VAR}[f'{col}_anomaly'] = np.abs({OUTPUT_VAR}[f'{col}_zscore']) > threshold

# 可视化结果
for col in columns:
    plt.figure(figsize=(15, 8))
    
    # 原始数据与移动平均线
    plt.subplot(2, 1, 1)
    plt.plot({OUTPUT_VAR}.index, {OUTPUT_VAR}[col], label='原始数据', alpha=0.6)
    plt.plot({OUTPUT_VAR}.index, {OUTPUT_VAR}[f'{col}_rolling_mean'], label=f'移动平均线 (窗口={window})', color='orange', linewidth=2)
    anomalies = {OUTPUT_VAR}[{OUTPUT_VAR}[f'{col}_anomaly']]
    plt.scatter(anomalies.index, anomalies[col], color='red', label='异常值', s=50, alpha=0.8)
    plt.title(f'移动窗口 Z-score 异常检测: {col}')
    plt.xlabel('时间' if isinstance({OUTPUT_VAR}.index, pd.DatetimeIndex) else '索引')
    plt.ylabel(col)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Z-score 与异常值标记
    plt.subplot(2, 1, 2)
    plt.plot({OUTPUT_VAR}.index, {OUTPUT_VAR}[f'{col}_zscore'], label='Z-score', color='blue')
    plt.axhline(y=threshold, color='red', linestyle='--', label=f'异常阈值 ({threshold})')
    plt.axhline(y=-threshold, color='red', linestyle='--')
    plt.scatter(anomalies.index, anomalies[f'{col}_zscore'], color='red', s=50, alpha=0.8, label='异常点')
    plt.title(f'移动窗口 Z-score 异常检测: Z-score 变化')
    plt.xlabel('时间' if isinstance({OUTPUT_VAR}.index, pd.DatetimeIndex) else '索引')
    plt.ylabel('Z-score')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
"""
)

# 7. 移动窗口 IQR 异常检测算法
moving_window_iqr_anomaly = Algorithm(
    id="anomaly_moving_window_iqr",
    name="移动窗口 IQR 异常检测",
    category="anomaly_detection",
    prompt="请使用移动窗口 IQR 方法检测 {VAR_NAME} 中的异常值。在滑动窗口内计算 IQR，将超出上下界的数据点标记为异常，并绘制原始数据与异常值标记图。",
    parameters=[
        AlgorithmParameter(name="columns", type="list", default=[], label="检测列", description="要检测异常值的列", widget="column-selector", priority="critical"),
        AlgorithmParameter(name="window", type="int", default=60, label="窗口大小", description="计算IQR的滑动窗口大小", min=10, max=300, step=10, priority="critical"),
        AlgorithmParameter(name="multiplier", type="float", default=1.5, label="IQR 倍数", description="IQR 上下界的倍数，通常为1.5或3.0", min=1.0, max=5.0, step=0.5, priority="critical"),
        AlgorithmParameter(name="center", type="bool", default=True, label="居中对齐", description="是否在窗口中居中对齐IQR", priority="non-critical")
    ],
    imports=["import numpy as np", "import pandas as pd", "import matplotlib.pyplot as plt"],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    template="""# Moving Window IQR Anomaly Detection for {VAR_NAME}
{OUTPUT_VAR} = {VAR_NAME}.copy()

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体显示中文
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

columns = {columns}
window = {window}
multiplier = {multiplier}
center = {center}

if not columns:
    columns = {OUTPUT_VAR}.select_dtypes(include=['number']).columns.tolist()

for col in columns:
    # 计算移动窗口的四分位数
    rolling_q1 = {OUTPUT_VAR}[col].rolling(window=window, center=center).quantile(0.25)
    rolling_q3 = {OUTPUT_VAR}[col].rolling(window=window, center=center).quantile(0.75)
    
    # 计算移动窗口IQR和上下界
    rolling_iqr = rolling_q3 - rolling_q1
    rolling_lower = rolling_q1 - multiplier * rolling_iqr
    rolling_upper = rolling_q3 + multiplier * rolling_iqr
    
    # 标记异常值
    {OUTPUT_VAR}[f'{col}_rolling_lower'] = rolling_lower
    {OUTPUT_VAR}[f'{col}_rolling_upper'] = rolling_upper
    {OUTPUT_VAR}[f'{col}_anomaly'] = ({OUTPUT_VAR}[col] < rolling_lower) | ({OUTPUT_VAR}[col] > rolling_upper)

# 可视化结果
for col in columns:
    plt.figure(figsize=(15, 8))
    
    # 原始数据与移动上下界
    plt.plot({OUTPUT_VAR}.index, {OUTPUT_VAR}[col], label='原始数据', alpha=0.6)
    plt.plot({OUTPUT_VAR}.index, {OUTPUT_VAR}[f'{col}_rolling_lower'], label=f'移动下界 ({multiplier}x IQR)', color='green', linestyle='--')
    plt.plot({OUTPUT_VAR}.index, {OUTPUT_VAR}[f'{col}_rolling_upper'], label=f'移动上界 ({multiplier}x IQR)', color='green', linestyle='--')
    anomalies = {OUTPUT_VAR}[{OUTPUT_VAR}[f'{col}_anomaly']]
    plt.scatter(anomalies.index, anomalies[col], color='red', label='异常值', s=50, alpha=0.8)
    plt.title(f'移动窗口 IQR 异常检测: {col} (窗口={window})')
    plt.xlabel('时间' if isinstance({OUTPUT_VAR}.index, pd.DatetimeIndex) else '索引')
    plt.ylabel(col)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
"""
)

algorithms = [threshold_sigma, isolation_forest, change_point, zscore_anomaly, iqr_anomaly, moving_window_zscore_anomaly, moving_window_iqr_anomaly]
