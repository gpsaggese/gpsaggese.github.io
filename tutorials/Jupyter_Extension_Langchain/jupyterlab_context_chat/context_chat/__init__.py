"""
JupyterLab Context Chat Server Extension.

This extension adds a /context-chat/* REST endpoint to Jupyter Server
for AI-powered chat functionality integrated with notebooks.
"""

import logging
from ._version import __version__
from .handlers import GenerateHandler

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _jupyter_labextension_paths():
    return [{
        "src": "labextension",
        "dest": "context-chat"
    }]


def _jupyter_server_extension_points():
    """
    Entry point for Jupyter Server extension discovery.
    
    Returns a list of extension point dictionaries.
    """
    return [{"module": "context_chat"}]


def _load_jupyter_server_extension(server_app):
    """
    Load the server extension.
    
    Registers the /context-chat/generate endpoint with the Jupyter web app.
    
    Args:
        server_app: The Jupyter Server application instance
    """
    web_app = server_app.web_app
    host_pattern = ".*"  # Match all hosts
    
    base_url = web_app.settings.get("base_url", "/")
    
    # Register the generate endpoint
    route_pattern = base_url + "context-chat/generate"
    
    handlers = [(route_pattern, GenerateHandler)]
    web_app.add_handlers(host_pattern, handlers)
    
    logger.info(f"Context Chat server extension loaded. Endpoint: {route_pattern}")
    logger.info("LangChain integration: Set OPENAI_API_KEY environment variable to enable AI responses")


# Legacy Jupyter Notebook server compatibility
_jupyter_server_extension_paths = _jupyter_server_extension_points
load_jupyter_server_extension = _load_jupyter_server_extension
