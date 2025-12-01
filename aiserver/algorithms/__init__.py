from .data_loading import algorithms as load_data_algos
from .data_operation import algorithms as data_operation_algos
from .data_preprocessing import algorithms as data_preprocessing_algos
from .eda import algorithms as eda_algos
from .anomaly_detection import algorithms as anomaly_detection_algos
from .trend import algorithms as trend_algos

all_algorithms = (
    load_data_algos +
    data_operation_algos +
    data_preprocessing_algos +
    eda_algos +
    anomaly_detection_algos +
    trend_algos
)

ALGORITHM_PARAMETERS = {}
ALGORITHM_TEMPLATES = {}
ALGORITHM_IMPORTS = {}

# Reconstruct dictionaries
for algo in all_algorithms:
    if algo.parameters:
        ALGORITHM_PARAMETERS[algo.id] = [p.to_dict() for p in algo.parameters]
    
    ALGORITHM_TEMPLATES[algo.id] = algo.template.strip() # Strip to be clean, though original might have newlines
    
    if algo.imports:
        ALGORITHM_IMPORTS[algo.id] = algo.imports

# Reconstruct ALGORITHM_PROMPTS
CATEGORY_LABELS = {
    "load_data": "输入输出",
    "data_operation": "数据操作",
    "data_preprocessing": "数据预处理",
    "eda": "探索式分析",
    "anomaly_detection": "异常检测",
    "trend_plot": "趋势绘制"
}

ALGORITHM_PROMPTS = {
    cat: {"label": label, "algorithms": []}
    for cat, label in CATEGORY_LABELS.items()
}

for algo in all_algorithms:
    if algo.category in ALGORITHM_PROMPTS:
        ALGORITHM_PROMPTS[algo.category]["algorithms"].append(algo.to_prompt_dict())
