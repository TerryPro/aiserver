import inspect
from typing import List, Dict, Any
from .base import Algorithm
from algorithm import (
    data_loading,
    data_operation,
    data_preprocessing,
    eda,
    anomaly_detection,
    trend,
    plotting,
    CATEGORY_LABELS
)

# Modules to scan for algorithms
MODULES = [
    data_loading,
    data_operation,
    data_preprocessing,
    eda,
    anomaly_detection,
    trend,
    plotting
]

all_algorithms: List[Algorithm] = []

# Scan modules and register algorithms
algorithms_dict = {}

for module in MODULES:
    for name, func in inspect.getmembers(module, inspect.isfunction):
        # Skip private functions
        if name.startswith('_'):
            continue
        
        # Try to create algorithm from function
        algo = Algorithm.from_func(func, module)
        if algo:
            # Avoid duplicates (e.g. aliases)
            if algo.id not in algorithms_dict:
                algorithms_dict[algo.id] = algo

all_algorithms = list(algorithms_dict.values())

# Initialize all algorithms (extract parameters and generate templates)
for algo in all_algorithms:
    try:
        algo.initialize()
    except Exception as e:
        print(f"Error initializing algorithm {algo.id}: {e}")
        import traceback
        traceback.print_exc()

ALGORITHM_PARAMETERS = {}
ALGORITHM_TEMPLATES = {}
ALGORITHM_IMPORTS = {}

# Reconstruct dictionaries
for algo in all_algorithms:
    if algo.parameters:
        ALGORITHM_PARAMETERS[algo.id] = [p.to_dict() for p in algo.parameters]
    
    ALGORITHM_TEMPLATES[algo.id] = algo.template.strip() if algo.template else ""
    
    if algo.imports:
        ALGORITHM_IMPORTS[algo.id] = algo.imports

# Reconstruct ALGORITHM_PROMPTS
ALGORITHM_PROMPTS = {
    cat: {"label": label, "algorithms": []}
    for cat, label in CATEGORY_LABELS.items()
}

for algo in all_algorithms:
    if algo.category in ALGORITHM_PROMPTS:
        ALGORITHM_PROMPTS[algo.category]["algorithms"].append(algo.to_prompt_dict())
