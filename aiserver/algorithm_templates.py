"""
Algorithm Code Templates
Contains code templates for various data analysis algorithms.
These templates are designed to be inserted into Jupyter Notebook cells.
Refactored to use aiserver.algorithms package
"""

from .algorithms import ALGORITHM_TEMPLATES, ALGORITHM_PARAMETERS, ALGORITHM_IMPORTS, all_algorithms, CATEGORY_LABELS
# 从workflow_lib导入所有算法函数
from .workflow_lib import *
import os

def get_library_metadata():
    """
    Constructs and returns the library metadata for the frontend.
    Returns:
        dict: A dictionary mapping category labels to lists of algorithm metadata.
    """
    library = {}
    
    # Pre-fetch csv files from dataset directory
    csv_files = _get_csv_files()
        
    for cat_id, label in CATEGORY_LABELS.items():
        library[label] = []
        
        # Find algorithms in this category
        category_algos = [algo for algo in all_algorithms if algo.category == cat_id]
        
        for algo in category_algos:
            # Convert Algorithm to dictionary for frontend
            args_list = _process_algorithm_parameters(algo, csv_files)

            # Use algorithm-defined ports (all algorithms now have explicit ports)
            ports_info = algo.to_port_dict()
            
            # Use pre-generated templates from Algorithm.initialize()
            template = algo.template
            
            # Determine node type
            node_type = "generic"
            if algo.id == "load_csv":
                node_type = "csv_loader"

            algo_dict = {
                "id": algo.id,
                "name": algo.name,
                "description": algo.prompt,
                "category": label,
                "template": template,
                "imports": algo.imports,
                "args": args_list,
                "inputs": ports_info["inputs"],
                "outputs": ports_info["outputs"],
                "nodeType": node_type
            }
            library[label].append(algo_dict)
            
    return library


def _get_csv_files():
    """Get CSV files from dataset directory."""
    dataset_path = os.path.join(os.getcwd(), 'dataset')
    try:
        return [f for f in os.listdir(dataset_path) if f.endswith('.csv')] if os.path.exists(dataset_path) else []
    except Exception:
        return []


def _process_algorithm_parameters(algo, csv_files):
    """Process algorithm parameters and populate file options."""
    args_list = []
    for p in algo.parameters:
        p_dict = p.to_dict()
        # Populate options for file-selector
        if p.widget == "file-selector" and not p_dict.get("options"):
            p_dict["options"] = csv_files
            # Set default if options available and default is generic
            if csv_files and (not p.default or "dataset/" in str(p.default)):
                p_dict["default"] = csv_files[0]
        args_list.append(p_dict)
    return args_list
