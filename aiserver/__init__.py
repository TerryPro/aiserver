from ._version import __version__
from .handlers import setup_handlers
import sys
import os

# Add library directory to sys.path to allow importing 'algorithm' without installation
# This supports dynamic editing of the algorithm library
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels to JuServer root (aiserver/aiserver -> aiserver -> JuServer)
    project_root = os.path.dirname(os.path.dirname(current_dir))
    library_path = os.path.join(project_root, "library")

    if library_path not in sys.path and os.path.exists(library_path):
        sys.path.insert(0, library_path)
        # print(f"Added {library_path} to sys.path")
except Exception as e:
    pass  # Fallback to installed package if path manipulation fails


def _jupyter_server_extension_points():
    return [{
        "module": "aiserver"
    }]


def _load_jupyter_server_extension(server_app):
    """Registers the API handler to receive HTTP requests from the frontend extension.

    Parameters
    ----------
    server_app: jupyterlab.labapp.LabApp
        JupyterLab application instance
    """
    setup_handlers(server_app.web_app)
    name = "aiserver"
    server_app.log.info(f"Registered {name} server extension")


# For backward compatibility with notebook server - useful for Binder/JupyterHub
load_jupyter_server_extension = _load_jupyter_server_extension
