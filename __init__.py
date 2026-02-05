"""
ComfyUI Flux MakeUp - Custom Node for AI-Powered Makeup Transfer

This module provides custom nodes for ComfyUI that enable makeup transfer
using the Flux model architecture. It includes nodes for loading models
and performing makeup sampling/generation.

Available Nodes:
- StableMakeup_LoadModel: Loads the Flux makeup model and face parsing network
- StableMakeup_Sampler: Performs makeup transfer between source and reference images
"""

import os
import sys

# Add the necessary paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
flux_makeup_dir = os.path.join(current_dir, "Flux_Makeup_ComfyUI")
custom_nodes_dir = os.path.join(flux_makeup_dir, "custom_nodes", "ComfyUI_Stable_Makeup")

# Add paths to sys.path if they exist
for path in [flux_makeup_dir, custom_nodes_dir]:
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

# Import the node mappings from the stable makeup nodes
try:
    from Flux_Makeup_ComfyUI.custom_nodes.ComfyUI_Stable_Makeup.stable_makeup_nodes import (
        NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS
    )
except ImportError:
    try:
        # Fallback: try direct import if running from within custom_nodes
        from stable_makeup_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    except ImportError:
        # If imports fail, provide empty mappings with a warning
        print("[Flux MakeUp] Warning: Could not import stable_makeup_nodes. Node mappings will be empty.")
        NODE_CLASS_MAPPINGS = {}
        NODE_DISPLAY_NAME_MAPPINGS = {}

# Web directory for any custom UI components (optional)
WEB_DIRECTORY = "./web"

# Version information
__version__ = "1.0.0"
__author__ = "Flux MakeUp Contributors"

# Export the required ComfyUI variables
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
    "__version__"
]
