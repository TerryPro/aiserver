
# Algorithm Parameter Registry
# Defines the parameters for each algorithm to enable UI generation

ALGORITHM_PARAMETERS = {
    "load_csv": [
        {
            "name": "filepath",
            "type": "str",
            "default": "dataset/data.csv",
            "label": "文件路径",
            "description": "CSV文件路径 (相对于项目根目录)",
            "widget": "file-selector"  # Hint for UI to render file selector
        }
    ],
    "import_variable": [
        {
            "name": "variable_name",
            "type": "str",
            "default": "",
            "label": "变量名称",
            "description": "当前会话中的DataFrame变量名",
            "widget": "variable-selector"
        }
    ],
    "smoothing_sg": [
        {
            "name": "window_length",
            "type": "int",
            "default": 11,
            "label": "窗口长度",
            "description": "必须是奇数且大于多项式阶数",
            "min": 3,
            "step": 2
        },
        {
            "name": "polyorder",
            "type": "int",
            "default": 3,
            "label": "多项式阶数",
            "description": "用于拟合样本的多项式阶数",
            "min": 1,
            "max": 5
        }
    ],
    "smoothing_ma": [
        {
            "name": "window_size",
            "type": "int",
            "default": 5,
            "label": "窗口大小",
            "description": "移动窗口的大小",
            "min": 1
        }
    ],
    "resampling_down": [
        {
            "name": "rule",
            "type": "str",
            "default": "1H",
            "label": "频率规则",
            "description": "目标频率 (例如 '1H', '1D', '15T')",
            "options": ["1T", "5T", "15T", "30T", "1H", "6H", "12H", "1D", "1W", "1M"]
        }
    ],
    "interpolation_spline": [
        {
            "name": "order",
            "type": "int",
            "default": 3,
            "label": "样条阶数",
            "description": "样条插值的阶数",
            "min": 1,
            "max": 5
        }
    ],
    "outlier_clip": [
        {
            "name": "lower_quantile",
            "type": "float",
            "default": 0.01,
            "label": "下分位数",
            "description": "裁剪下限 (0.0-1.0)",
            "min": 0.0,
            "max": 0.5,
            "step": 0.01
        },
        {
            "name": "upper_quantile",
            "type": "float",
            "default": 0.99,
            "label": "上分位数",
            "description": "裁剪上限 (0.0-1.0)",
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
            "label": "最大滞后阶数",
            "description": "生成的最大滞后阶数",
            "min": 1,
            "max": 24
        }
    ],
    "autocorrelation": [
        {
            "name": "lags",
            "type": "int",
            "default": 50,
            "label": "滞后数",
            "description": "绘制的滞后数量",
            "min": 10,
            "max": 200
        }
    ],
    "isolation_forest": [
        {
            "name": "contamination",
            "type": "float",
            "default": 0.05,
            "label": "污染率",
            "description": "预期的异常值比例",
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
            "label": "图表类型",
            "options": ["line", "bar", "scatter"],
            "description": "图表的类型"
        },
        {
            "name": "column",
            "type": "str",
            "default": "",
            "label": "列名",
            "description": "要绘制的列 (Y轴)"
        }
    ],
    "merge_dfs": [
        {
            "name": "how",
            "type": "str",
            "default": "inner",
            "label": "合并方式",
            "options": ["inner", "outer", "left", "right"],
            "description": "执行合并的方式"
        },
        {
            "name": "on",
            "type": "str",
            "default": "",
            "label": "合并列",
            "description": "用于连接的列名或索引级别名。留空则使用索引。",
            "widget": "column-selector"
        }
    ],
    "train_test_split": [
        {
            "name": "test_size",
            "type": "float",
            "default": 0.2,
            "label": "测试集比例",
            "description": "包含在测试拆分中的数据集比例",
            "min": 0.01,
            "max": 0.99,
            "step": 0.05
        },
        {
            "name": "target_column",
            "type": "str",
            "default": "target",
            "label": "目标列",
            "description": "目标变量列名 (y)",
            "widget": "column-selector"
        },
        {
            "name": "random_state",
            "type": "int",
            "default": 42,
            "label": "随机种子",
            "description": "控制拆分前的数据打乱"
        }
    ],
    "trend_plot": [
        {"name": "x_column", "type": "str", "default": "", "label": "X轴列名", "description": "作为X轴的列 (留空则使用索引)", "widget": "column-selector"},
        {"name": "y_columns", "type": "list", "default": [], "label": "Y轴列名", "description": "Y轴数据列 (留空则绘制所有数值列)", "widget": "column-selector"},
        {"name": "title", "type": "str", "default": "趋势图", "label": "图表标题", "description": "图表的标题"},
        {"name": "xlabel", "type": "str", "default": "", "label": "X轴标签", "description": "X轴的显示标签"},
        {"name": "ylabel", "type": "str", "default": "", "label": "Y轴标签", "description": "Y轴的显示标签"},
        {"name": "grid", "type": "bool", "default": True, "label": "显示网格", "description": "是否显示背景网格"},
        {"name": "figsize", "type": "str", "default": "(10, 6)", "label": "图像尺寸", "description": "图像大小元组，例如 (10, 6)"}
    ],
    "select_columns": [
        {"name": "columns", "type": "list", "default": [], "label": "选择列", "description": "要选择的列列表", "widget": "column-selector"}
    ],
    "filter_rows": [
        {"name": "condition", "type": "str", "default": "", "label": "过滤条件", "description": "查询字符串 (例如 'age > 18')"}
    ],
    "sort_values": [
        {"name": "by", "type": "list", "default": [], "label": "排序依据", "description": "排序的依据列", "widget": "column-selector"},
        {"name": "ascending", "type": "bool", "default": True, "label": "升序", "description": "升序还是降序"}
    ],
    "groupby_agg": [
        {"name": "by", "type": "list", "default": [], "label": "分组依据", "description": "分组的依据列", "widget": "column-selector"},
        {"name": "agg_dict", "type": "dict", "default": {}, "label": "聚合字典", "description": "聚合配置字典 (例如 {'col': 'mean'})"}
    ],
    "pivot_table": [
        {"name": "values", "type": "str", "default": "", "label": "值列", "description": "要聚合的列", "widget": "column-selector"},
        {"name": "index", "type": "list", "default": [], "label": "索引列", "description": "索引列列表", "widget": "column-selector"},
        {"name": "columns", "type": "list", "default": [], "label": "列名列", "description": "列名列列表", "widget": "column-selector"},
        {"name": "aggfunc", "type": "str", "default": "mean", "label": "聚合函数", "description": "聚合函数"}
    ],
    "concat_dfs": [
        {"name": "axis", "type": "int", "default": 0, "label": "轴向", "description": "拼接轴向 (0=行, 1=列)"}
    ],
    "rename_columns": [
        {"name": "columns_map", "type": "dict", "default": {}, "label": "列名映射", "description": "旧名到新名的映射字典"}
    ],
    "drop_duplicates": [
        {"name": "subset", "type": "list", "default": [], "label": "子集", "description": "考虑重复的列子集", "widget": "column-selector"},
        {"name": "keep", "type": "str", "default": "first", "label": "保留策略", "options": ["first", "last", "False"], "description": "保留哪个重复项"}
    ],
    "fill_na": [
        {"name": "value", "type": "str", "default": None, "label": "填充值", "description": "用于填充的常数值 (可选)"},
        {"name": "method", "type": "enum", "default": None, "label": "填充方法", "options": ["ffill", "bfill"], "description": "填充方法 (可选)"}
    ],
    "export_data": [
        {"name": "global_name", "type": "str", "default": "exported_data", "label": "全局变量名", "description": "引出的全局变量名称"}
    ],
    "astype": [
        {"name": "dtype_map", "type": "dict", "default": {}, "label": "类型映射", "description": "列到类型的映射字典"}
    ],
    "apply_func": [
        {"name": "func_code", "type": "str", "default": "lambda x: x", "label": "函数代码", "description": "函数或lambda的Python代码"},
        {"name": "axis", "type": "int", "default": 0, "label": "轴向", "description": "应用轴向 (0或1)"}
    ]
}
