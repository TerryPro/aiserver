
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
                "prompt": "请对数据进行平滑处理。使用 scipy.signal.savgol_filter 算法，目的是去除高频噪声并保留波形特征。请根据数据特性自动选择或建议合适的窗口长度和多项式阶数。"
            },
            {
                "id": "smoothing_ma",
                "name": "移动平均平滑",
                "prompt": "请对数据进行移动平均平滑处理。使用 pandas 的 rolling().mean() 方法。请根据数据采样频率建议一个合理的窗口大小。"
            },
            {
                "id": "interpolation_time",
                "name": "时间加权插值",
                "prompt": "对时序数据的缺失值进行插值。由于数据是不等间隔的，请使用 pandas 的 interpolate(method='time') 进行时间加权插值，以确保物理意义的准确性。"
            },
            {
                "id": "interpolation_spline",
                "name": "样条插值",
                "prompt": "对时序数据进行样条插值 (Spline Interpolation)。使用 pandas 的 interpolate(method='spline', order=3) 方法，以获得更平滑的补全曲线。"
            },
            {
                "id": "resampling_down",
                "name": "降采样 (聚合)",
                "prompt": "对高频数据进行降采样处理。使用 pandas 的 resample() 方法，将数据聚合到更低的时间分辨率（例如 '1min' 或 '1H'）。对于数值列使用 mean() 聚合，对于状态列使用 last() 或 max()。"
            },
            {
                "id": "alignment",
                "name": "多源数据对齐",
                "prompt": "对多个不同采样频率的时序数据进行时间对齐。使用 pandas 的 merge_asof 方法，以主时间轴为基准，将其他数据对齐到该时间轴上。"
            }
        ]
    },
    "eda": {
        "label": "探索式分析",
        "algorithms": [
            {
                "id": "summary_stats",
                "name": "时序统计摘要",
                "prompt": "请计算时序数据的详细统计摘要。除了基本的 describe() 之外，请计算偏度 (skewness)、峰度 (kurtosis)，并统计时间范围、采样间隔和缺失率。"
            },
            {
                "id": "line_plot",
                "name": "多尺度时序曲线",
                "prompt": "绘制专业的时序曲线图。使用 matplotlib 或 seaborn。要求：1. X轴格式化为时间显示；2. 支持缩放；3. 添加网格、图例和中文标签；4. 如果数据量过大，请先进行降采样再绘图。"
            },
            {
                "id": "spectral_analysis",
                "name": "频谱分析 (PSD)",
                "prompt": "请进行频域分析。计算并绘制信号的功率谱密度 (PSD)。使用 scipy.signal.welch 方法，并以对数坐标显示频率和功率。这有助于识别信号中的周期性成分。"
            },
            {
                "id": "autocorrelation",
                "name": "自相关分析 (ACF)",
                "prompt": "进行自相关分析。计算并绘制自相关函数 (ACF) 图，使用 statsmodels.graphics.tsaplots.plot_acf。这有助于发现数据的周期性模式。"
            },
            {
                "id": "decomposition",
                "name": "STL 分解",
                "prompt": "对时序数据进行 STL 分解 (Seasonal-Trend decomposition using LOESS)。将数据分解为趋势项、季节项和残差项，并绘制分解结果图。"
            },
            {
                "id": "heatmap_distribution",
                "name": "时序热力图",
                "prompt": "绘制时序热力图以观察数据的分布模式。例如，X轴为日期，Y轴为小时，颜色表示数值大小。这有助于发现日内模式或季节性变化。"
            }
        ]
    },
    "anomaly_detection": {
        "label": "异常检测",
        "algorithms": [
            {
                "id": "threshold_sigma",
                "name": "3-Sigma 阈值检测",
                "prompt": "使用 3-Sigma 原则检测异常值。计算移动窗口内的均值和标准差，将超过 mean ± 3*std 的点标记为异常。请在原图上用红色标记出异常点。"
            },
            {
                "id": "isolation_forest",
                "name": "孤立森林检测",
                "prompt": "使用孤立森林 (Isolation Forest) 算法检测时序数据中的异常点。请设置污染率 (contamination) 参数，并在原图上用红色标记出识别出的异常点。"
            },
            {
                "id": "change_point",
                "name": "变点检测",
                "prompt": "检测时序数据的统计特性发生突变的时间点 (Change Point Detection)。可以使用 ruptures 库或基于滑动窗口的统计差异检测方法。"
            }
        ]
    }
}
