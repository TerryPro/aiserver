import inspect
import pkgutil
import importlib
from typing import List, Dict, Any
from .base import Algorithm
import algorithm

# Import CATEGORY_LABELS
try:
    from algorithm import CATEGORY_LABELS
except ImportError:
    print("Warning: CATEGORY_LABELS not found in algorithm package")
    CATEGORY_LABELS = {}

all_algorithms: List[Algorithm] = []
algorithms_dict = {}

def get_all_modules(package):
    """Recursively find all modules in a package."""
    modules = []
    if hasattr(package, "__path__"):
        for _, name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
            if not is_pkg:
                try:
                    module = importlib.import_module(name)
                    modules.append(module)
                except Exception as e:
                    print(f"Failed to import {name}: {e}")
    return modules

# Scan modules
modules = get_all_modules(algorithm)

for module in modules:
    for name, func in inspect.getmembers(module, inspect.isfunction):
        if name.startswith('_'): continue
        
        # Try to create algorithm
        algo = Algorithm.from_func(func, module)
        if algo:
            # Check if this function is actually defined in this module
            # This prevents registering the same algorithm multiple times if it is imported in other modules
            if func.__module__ == module.__name__:
                if algo.id not in algorithms_dict:
                    algorithms_dict[algo.id] = algo
                else:
                    # Already registered
                    pass

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
    # Handle category mapping if needed (e.g. if we changed category IDs in files)
    # The migration script updated "数据绘图" to "plotting" in docstrings
    # And "CATEGORY_LABELS" has "plotting".
    
    if algo.category in ALGORITHM_PROMPTS:
        ALGORITHM_PROMPTS[algo.category]["algorithms"].append(algo.to_prompt_dict())
    else:
        # Fallback for unknown categories
        if algo.category not in ALGORITHM_PROMPTS:
             ALGORITHM_PROMPTS[algo.category] = {"label": algo.category, "algorithms": []}
        ALGORITHM_PROMPTS[algo.category]["algorithms"].append(algo.to_prompt_dict())
