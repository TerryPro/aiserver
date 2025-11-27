
"""
Algorithm Prompt Library
Contains structured prompts for various data analysis algorithms.
"""

ALGORITHM_PROMPTS = {
    "data_preprocessing": {
        "label": "数据预处理",
        "algorithms": [
            {
                "id": "smoothing_sg",
                "name": "Savitzky-Golay 平滑",
                "prompt": "请对{VAR_NAME} 执行 Savitzky-Golay 平滑处理。使用 scipy.signal.savgol_filter 去除高频噪声并保留波形特征，并根据数据特性选择或建议合适的窗口长度与多项式阶数。"
            },
            {
                "id": "smoothing_ma",
                "name": "移动平均平滑",
                "prompt": "请对{VAR_NAME} 执行移动平均平滑。使用 pandas 的 rolling().mean() 方法，并根据采样频率选择合理的窗口大小。"
            },
            {
                "id": "interpolation_time",
                "name": "时间加权插值",
                "prompt": "请对{VAR_NAME} 的缺失值进行时间加权插值。不等间隔数据使用 pandas 的 interpolate(method='time') 以确保物理意义的准确性。"
            },
            {
                "id": "interpolation_spline",
                "name": "样条插值",
                "prompt": "请对{VAR_NAME} 进行样条插值 (Spline)。使用 pandas 的 interpolate(method='spline', order=3) 以获得更平滑的补全曲线。"
            },
            {
                "id": "resampling_down",
                "name": "降采样 (聚合)",
                "prompt": "请对{VAR_NAME} 进行降采样聚合。使用 pandas 的 resample() 将数据聚合到更低的时间分辨率（例如 '1min' 或 '1H'）；数值列使用 mean()，状态列使用 last() 或 max()。"
            },
            {
                "id": "alignment",
                "name": "多源数据对齐",
                "prompt": "请以 {VAR_NAME} 为基准执行多源时间对齐。使用 pandas 的 merge_asof 方法，将其他数据对齐到该时间轴。"
            }
        ]
    },
    "eda": {
        "label": "探索式分析",
        "algorithms": [
            {
                "id": "summary_stats",
                "name": "时序统计摘要",
                "prompt": "请对{VAR_NAME} 执行时序统计摘要分析。\n代码应包含以下步骤:\n1. 基础统计量：使用 pandas 的 describe() 方法计算均值、标准差、最小值、最大值及四分位数。\n2. 分布特征：计算数据的偏度 (skewness) 和峰度 (kurtosis) 以评估数据分布形态。\n3. 时间特性分析：\n   - 计算时间覆盖范围（起始时间、结束时间、总时长）。\n   - 推断或计算平均采样间隔/频率。\n4. 数据质量评估：统计缺失值的数量及占比。\n请将所有结果汇总为一个清晰的 DataFrame 或字典格式返回，以便于查看。"
            },
            {
                "id": "line_plot",
                "name": "多尺度时序曲线",
                "prompt": "请对{VAR_NAME} 绘制时序曲线图。使用 matplotlib 或 seaborn，并满足：1. X轴时间格式化；2. 支持缩放；3. 添加网格、图例与中文标签；4. 数据量过大时先降采样再绘图。"
            },
            {
                "id": "spectral_analysis",
                "name": "频谱分析 (PSD)",
                "prompt": "请对{VAR_NAME} 进行频域分析，计算并绘制功率谱密度 (PSD)。使用 scipy.signal.welch，并以对数坐标显示频率和功率，用于识别周期性成分。"
            },
            {
                "id": "autocorrelation",
                "name": "自相关分析 (ACF)",
                "prompt": "请对{VAR_NAME} 进行自相关分析。计算并绘制 ACF 图，使用 statsmodels.graphics.tsaplots.plot_acf，以发现周期性模式。"
            },
            {
                "id": "decomposition",
                "name": "STL 分解",
                "prompt": "请对{VAR_NAME} 执行 STL 分解 (Seasonal-Trend decomposition using LOESS)。将数据分解为趋势、季节与残差，并绘制分解结果图。"
            },
            {
                "id": "heatmap_distribution",
                "name": "时序热力图",
                "prompt": "请基于 {VAR_NAME} 绘制时序热力图，用于观察分布模式（例如 X=日期，Y=小时，颜色=数值）。用于发现日内模式或季节性变化。"
            }
        ]
    },
    "anomaly_detection": {
        "label": "异常检测",
        "algorithms": [
            {
                "id": "threshold_sigma",
                "name": "3-Sigma 阈值检测",
                "prompt": "请在 {VAR_NAME} 上应用 3-Sigma 异常检测。计算移动窗口均值与标准差，将超过 mean ± 3*std 的点标记为异常，并在原图用红色标记异常点。"
            },
            {
                "id": "isolation_forest",
                "name": "孤立森林检测",
                "prompt": "请对 {VAR_NAME} 执行孤立森林 (Isolation Forest) 异常检测。设置污染率 (contamination) 参数，并在原图用红色标记识别出的异常点。"
            },
            {
                "id": "change_point",
                "name": "变点检测",
                "prompt": "请在 {VAR_NAME} 中检测统计特性发生突变的时间点 (Change Point Detection)。可以使用 ruptures 库或基于滑动窗口的统计差异检测方法。"
            }
        ]
    }
}
