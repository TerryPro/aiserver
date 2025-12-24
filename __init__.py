"""
AI Server Extension for JupyterLab
"""
from aiserver.aiserver import (
    _jupyter_server_extension_points,
    _load_jupyter_server_extension,
    load_jupyter_server_extension,
)

from aiserver.aiserver.tablewalker import show_df

__all__ = [
    "_jupyter_server_extension_points",
    "_load_jupyter_server_extension", 
    "load_jupyter_server_extension",
    "show_df",
]