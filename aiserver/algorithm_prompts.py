
"""
Algorithm Prompt Library
Contains structured prompts for various data analysis algorithms.
"""

ALGORITHM_PROMPTS = {
    "load_data": {
        "label": "加载数据",
        "algorithms": [
            {
                "id": "load_csv",
                "name": "加载 CSV",
                "prompt": "请加载 CSV 文件。使用 pandas.read_csv 读取指定路径的文件，并显示前几行数据。"
            },
            {
                "id": "import_variable",
                "name": "引入变量",
                "prompt": "请引入已存在的 DataFrame 变量 {variable_name}。创建其副本或引用，以便后续分析使用。"
            }
        ]
    },
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
            },
            {
                "id": "feature_scaling",
                "name": "数据标准化/归一化",
                "prompt": "请对 {VAR_NAME} 进行特征缩放。提供 Z-Score 标准化和 Min-Max 归一化两种结果，便于后续模型训练。"
            },
            {
                "id": "diff_transform",
                "name": "差分变换",
                "prompt": "请对 {VAR_NAME} 进行一阶和二阶差分处理，以消除趋势并使数据平稳，绘制差分后的时序图。"
            },
            {
                "id": "outlier_clip",
                "name": "离群值盖帽 (Winsorization)",
                "prompt": "请对 {VAR_NAME} 进行离群值盖帽处理。将超出 1% 和 99% 分位数的值限制在边界范围内，以减少极端值的影响。"
            },
            {
                "id": "feature_extraction_time",
                "name": "时间特征提取",
                "prompt": "请从 {VAR_NAME} 的时间索引中提取特征。生成‘小时’、‘星期几’、‘月份’、‘是否周末’等新列，用于机器学习模型输入。"
            },
            {
                "id": "feature_lag",
                "name": "滞后特征生成",
                "prompt": "请对 {VAR_NAME} 生成滞后特征。创建滞后 1 至 3 个时间步的列（lag_1, lag_2, lag_3），用于自回归分析。"
            },
            {
                "id": "transform_log",
                "name": "对数变换",
                "prompt": "请对 {VAR_NAME} 进行对数变换。使用 log1p 处理以稳定方差，并绘制变换前后的分布对比图。"
            },
            {
                "id": "filter_butterworth",
                "name": "巴特沃斯低通滤波",
                "prompt": "请对 {VAR_NAME} 应用巴特沃斯低通滤波器。设置截止频率和阶数，去除高频噪声，保留主要趋势信号。"
            },
            {
                "id": "merge_dfs",
                "name": "数据合并 (Merge)",
                "prompt": "请合并两个数据框 {left} 和 {right}。根据指定的合并方式（inner, outer, left, right）和连接键进行 pd.merge 操作。"
            },
            {
                "id": "train_test_split",
                "name": "训练/测试集分割",
                "prompt": "请将 {data} 分割为训练集和测试集。使用 sklearn.model_selection.train_test_split，返回 X_train, X_test, y_train, y_test。"
            }
        ]
    },
    "eda": {
        "label": "探索式分析",
        "algorithms": [
            {
                "id": "plot_custom",
                "name": "通用绘图 (Plot)",
                "prompt": "请对 {VAR_NAME} 进行绘图。根据用户选择的图表类型（折线图、柱状图、散点图）和指定列进行可视化。"
            },
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
    },
    "trend_plot": {
        "label": "趋势绘制",
        "algorithms": [
            {
                "id": "trend_plot",
                "name": "通用趋势图 (Trend)",
                "prompt": "请根据配置绘制 {VAR_NAME} 的趋势图。支持自定义 X 轴、Y 轴列、标题、网格等设置。"
            },
            {
                "id": "trend_ma",
                "name": "移动平均趋势",
                "prompt": "请对{VAR_NAME} 绘制移动平均趋势线。先推断采样频率并将数据重采样到统一时间轴（如 '1S'），选择合理的窗口长度（例如 60 或 300 秒），使用 pandas 的 rolling().mean() 计算趋势线，并用 matplotlib 绘制原始曲线与趋势线，添加网格、图例与中文标签。若 {VAR_NAME} 为 DataFrame，请对数值列分别绘制。"
            },
            {
                "id": "trend_ewma",
                "name": "指数加权趋势",
                "prompt": "请对{VAR_NAME} 绘制 EWMA（指数加权移动平均）趋势线。统一时间轴后，依据采样频率选择合适的 span（如 60 或 300），使用 pandas 的 ewm(span=...).mean() 计算趋势，并使用 matplotlib 将原始数据与 EWMA 趋势曲线叠加展示。"
            },
            {
                "id": "trend_loess",
                "name": "LOESS 趋势",
                "prompt": "请对{VAR_NAME} 绘制 LOESS 平滑趋势。将时间序列统一到同一采样频率后，使用 statsmodels.nonparametric.smoothers_lowess.lowess 进行平滑并绘制趋势曲线；若缺少该库，可退化为 rolling().mean()。图表需包含中文标题、轴标签与图例。"
            },
            {
                "id": "trend_polyfit",
                "name": "多项式趋势拟合",
                "prompt": "请对{VAR_NAME} 进行多项式趋势拟合并绘制趋势。将时间戳转换为连续时间序列（秒或索引），使用 numpy.polyfit 对 1~2 阶进行拟合，绘制拟合曲线与原始数据，并计算与输出拟合优度（R²）。"
            },
            {
                "id": "trend_stl_trend",
                "name": "STL 趋势分量",
                "prompt": "请对{VAR_NAME} 执行 STL 分解并提取趋势分量。统一采样频率后，使用 statsmodels.tsa.seasonal.STL 提取趋势，绘制趋势曲线并与原始数据对比显示；根据卫星遥测的特性选择合适的季节周期（如日照周期）。"
            },
            {
                "id": "trend_basic_stacked",
                "name": "基础趋势绘制（分栏）",
                "prompt": "请按照原始样式对 {VAR_NAME} 进行趋势绘制（分栏布局）。每个数值列单独占一行子图，统一时间轴。实现要点：\n1) 推断采样频率并重采样为统一时间轴（如 '1S'）；\n2) 仅对数值列绘图，数据量大时先降采样（如 1S/5S 或 rolling）；\n3) 使用 matplotlib，添加中文标题、轴标签、网格与图例；\n4) 若 {VAR_NAME} 为 Series，则直接在单个子图绘制。"
            },
            {
                "id": "trend_basic_overlay",
                "name": "基础趋势绘制（叠加）",
                "prompt": "请按照原始样式对 {VAR_NAME} 进行趋势绘制（叠加布局）。所有数值列绘制在同一坐标轴上并区分图例，统一时间轴。实现要点：\n1) 推断采样频率并重采样为统一时间轴（如 '1S'）；\n2) 仅对数值列绘图，数据量大时先降采样（如 1S/5S 或 rolling）；\n3) 使用 matplotlib，添加中文标题、轴标签、网格与图例；\n4) 若 {VAR_NAME} 为 Series，则直接在单图叠加绘制（仅一条曲线）。"
            },
            {
                "id": "trend_basic_grid",
                "name": "基础趋势绘制（网格）",
                "prompt": "请按照原始样式对 {VAR_NAME} 进行趋势绘制（网格布局）。根据列数量自动计算行列数形成子图网格（如 2xN 或近似方阵），统一时间轴。实现要点：\n1) 推断采样频率并重采样为统一时间轴（如 '1S'）；\n2) 仅对数值列绘图，数据量大时先降采样（如 1S/5S 或 rolling）；\n3) 使用 matplotlib，添加中文标题、轴标签、网格与图例；\n4) 若 {VAR_NAME} 为 Series，则在单个子图绘制。"
            }
        ]
    }
}
