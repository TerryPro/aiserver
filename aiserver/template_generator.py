from typing import List, Dict, Any

def generate_full_template(algo_name: str, source: str) -> str:
    """
    Generate simplified code template (only source).
    
    Args:
        algo_name: Name of the algorithm (for comments)
        source: Source code of the function
    """
    return f"# {algo_name}\n{source}"

