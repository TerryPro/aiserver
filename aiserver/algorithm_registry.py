
# Algorithm Parameter Registry
# Defines the parameters for each algorithm to enable UI generation

ALGORITHM_PARAMETERS = {
    "load_csv": [
        {
            "name": "filepath",
            "type": "str",
            "default": "dataset/data.csv",
            "label": "File Path",
            "description": "Path to the CSV file (relative to project root)",
            "widget": "file-selector"  # Hint for UI to render file selector
        }
    ],
    "import_variable": [
        {
            "name": "variable_name",
            "type": "str",
            "default": "",
            "label": "Variable Name",
            "description": "Name of the DataFrame variable in the current session",
            "widget": "variable-selector"
        }
    ],
    "smoothing_sg": [
        {
            "name": "window_length",
            "type": "int",
            "default": 11,
            "label": "Window Length",
            "description": "Must be odd and greater than polyorder",
            "min": 3,
            "step": 2
        },
        {
            "name": "polyorder",
            "type": "int",
            "default": 3,
            "label": "Polynomial Order",
            "description": "The order of the polynomial used to fit the samples",
            "min": 1,
            "max": 5
        }
    ],
    "smoothing_ma": [
        {
            "name": "window_size",
            "type": "int",
            "default": 5,
            "label": "Window Size",
            "description": "Size of the moving window",
            "min": 1
        }
    ],
    "resampling_down": [
        {
            "name": "rule",
            "type": "str",
            "default": "1H",
            "label": "Frequency Rule",
            "description": "Target frequency (e.g., '1H', '1D', '15T')",
            "options": ["1T", "5T", "15T", "30T", "1H", "6H", "12H", "1D", "1W", "1M"]
        }
    ],
    "interpolation_spline": [
        {
            "name": "order",
            "type": "int",
            "default": 3,
            "label": "Spline Order",
            "description": "Order of the spline interpolation",
            "min": 1,
            "max": 5
        }
    ],
    "outlier_clip": [
        {
            "name": "lower_quantile",
            "type": "float",
            "default": 0.01,
            "label": "Lower Quantile",
            "description": "Lower bound for clipping (0.0-1.0)",
            "min": 0.0,
            "max": 0.5,
            "step": 0.01
        },
        {
            "name": "upper_quantile",
            "type": "float",
            "default": 0.99,
            "label": "Upper Quantile",
            "description": "Upper bound for clipping (0.0-1.0)",
            "min": 0.5,
            "max": 1.0,
            "step": 0.01
        }
    ],
    "feature_lag": [
        {
            "name": "max_lag",
            "type": "int",
            "default": 3,
            "label": "Max Lag Steps",
            "description": "Maximum number of lag steps to generate",
            "min": 1,
            "max": 24
        }
    ],
    "autocorrelation": [
        {
            "name": "lags",
            "type": "int",
            "default": 50,
            "label": "Lags",
            "description": "Number of lags to plot",
            "min": 10,
            "max": 200
        }
    ],
    "isolation_forest": [
        {
            "name": "contamination",
            "type": "float",
            "default": 0.05,
            "label": "Contamination",
            "description": "Expected proportion of outliers",
            "min": 0.001,
            "max": 0.5,
            "step": 0.01
        }
    ],
    "plot_custom": [
        {
            "name": "plot_type",
            "type": "str",
            "default": "line",
            "label": "Plot Type",
            "options": ["line", "bar", "scatter"],
            "description": "Type of the plot"
        },
        {
            "name": "column",
            "type": "str",
            "default": "",
            "label": "Column",
            "description": "Column to plot (for Y axis)"
        }
    ],
    "merge_dfs": [
        {
            "name": "how",
            "type": "str",
            "default": "inner",
            "label": "How",
            "options": ["inner", "outer", "left", "right"],
            "description": "Type of merge to be performed"
        },
        {
            "name": "on",
            "type": "str",
            "default": "",
            "label": "On (Column)",
            "description": "Column or index level names to join on. Leave empty to use index."
        }
    ],
    "train_test_split": [
        {
            "name": "test_size",
            "type": "float",
            "default": 0.2,
            "label": "Test Size",
            "description": "Proportion of the dataset to include in the test split",
            "min": 0.01,
            "max": 0.99,
            "step": 0.05
        },
        {
            "name": "target_column",
            "type": "str",
            "default": "target",
            "label": "Target Column",
            "description": "Name of the target variable column (y)"
        },
        {
            "name": "random_state",
            "type": "int",
            "default": 42,
            "label": "Random State",
            "description": "Controls the shuffling applied to the data before applying the split"
        }
    ],
    "trend_plot": [
        {"name": "x_column", "type": "str", "default": "", "label": "X轴列名", "description": "作为X轴的列 (留空则使用索引)"},
        {"name": "y_columns", "type": "str", "default": "", "label": "Y轴列名", "description": "Y轴数据列，用逗号分隔 (留空则绘制所有数值列)"},
        {"name": "title", "type": "str", "default": "趋势图", "label": "图表标题", "description": "图表的标题"},
        {"name": "xlabel", "type": "str", "default": "", "label": "X轴标签", "description": "X轴的显示标签"},
        {"name": "ylabel", "type": "str", "default": "", "label": "Y轴标签", "description": "Y轴的显示标签"},
        {"name": "grid", "type": "bool", "default": True, "label": "显示网格", "description": "是否显示背景网格"},
        {"name": "figsize", "type": "str", "default": "(10, 6)", "label": "图像尺寸", "description": "图像大小元组，例如 (10, 6)"}
    ]
}
