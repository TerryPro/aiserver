"""
Algorithm Code Templates
Contains code templates for various data analysis algorithms.
These templates are designed to be inserted into Jupyter Notebook cells.
Refactored to use aiserver.algorithms package
"""

from .algorithms import ALGORITHM_TEMPLATES, ALGORITHM_IMPORTS, all_algorithms, CATEGORY_LABELS
import os

def get_library_metadata():
    """
    Constructs and returns the library metadata for the frontend.
    Returns:
        dict: A dictionary mapping category labels to lists of algorithm metadata.
    """
    library = {}
    
    # Pre-fetch csv files from dataset directory
    csv_files = []
    dataset_path = os.path.join(os.getcwd(), 'dataset')
    if os.path.exists(dataset_path):
        try:
            csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
        except Exception as e:
            print(f"Error listing dataset directory: {e}")
            
    for cat_id, label in CATEGORY_LABELS.items():
        library[label] = []
        
        # Find algorithms in this category
        category_algos = [algo for algo in all_algorithms if algo.category == cat_id]
        
        for algo in category_algos:
            # Convert Algorithm to dictionary for frontend
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

            algo_dict = {
                "id": algo.id,
                "name": algo.name,
                "description": algo.prompt, # Using prompt as description
                "category": label,
                "template": algo.template,
                "imports": algo.imports,
                "args": args_list
            }
            library[label].append(algo_dict)
            
    return library
