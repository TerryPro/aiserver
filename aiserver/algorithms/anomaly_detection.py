from .base import Algorithm, AlgorithmParameter, Port

threshold_sigma = Algorithm(
    id="threshold_sigma",
    name="3-Sigma 阈值检测",
    category="anomaly_detection",
    prompt="请在 {VAR_NAME} 上应用 3-Sigma 异常检测。计算移动窗口均值与标准差，将超过 mean ± 3*std 的点标记为异常，并在原图用红色标记异常点。",
    parameters=[],
    imports=["import pandas as pd", "import matplotlib.pyplot as plt"],
    inputs=[Port(name="df_in")],
    outputs=[Port(name="df_out")],
    template="""
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

# Detection
algo = rpt.Binseg(model="l2").fit(signal)
result = algo.predict(n_bkps=5) # Detect 5 change points

# Display
rpt.display(signal, result, figsize=(15, 6))
plt.title(f'Change Point Detection: {target_col}')
plt.show()
"""
)

algorithms = [threshold_sigma, isolation_forest, change_point]
