"""
ComfyUI Flux MakeUp - Custom Node for AI-Powered Makeup Transfer

This module provides custom nodes for ComfyUI that enable makeup transfer
using the Flux model architecture. It includes nodes for loading models
and performing makeup sampling/generation.

Available Nodes:
- StableMakeup_LoadModel: Loads the Flux makeup model and face parsing network
- StableMakeup_Sampler: Performs makeup transfer between source and reference images

Installation:
    Place this folder in ComfyUI's custom_nodes directory.
    The module will automatically register the nodes on ComfyUI startup.

Troubleshooting:
    If nodes don't appear, check that:
    1. The folder is in custom_nodes/ directory
    2. All required dependencies are installed
    3. ComfyUI is restarted after installation
"""

import os
import sys
import logging

# Set up logging for debugging
logger = logging.getLogger(__name__)

# Add the necessary paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
flux_makeup_dir = os.path.join(current_dir, "Flux_Makeup_ComfyUI")
custom_nodes_dir = os.path.join(flux_makeup_dir, "custom_nodes", "ComfyUI_Stable_Makeup")

# Add paths to sys.path if they exist (required for dependency resolution)
for path in [flux_makeup_dir, custom_nodes_dir, current_dir]:
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

# Initialize empty mappings as fallback
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def _load_node_mappings():
    """
    Load node class mappings from the stable_makeup_nodes module.
    
    This function attempts multiple import strategies to ensure compatibility
    with different installation scenarios (direct install, symlink, etc.)
    """
    global NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    
    import_errors = []
    
    # Strategy 1: Full path import from nested structure
    try:
        from Flux_Makeup_ComfyUI.custom_nodes.ComfyUI_Stable_Makeup.stable_makeup_nodes import (
            NODE_CLASS_MAPPINGS as mappings,
            NODE_DISPLAY_NAME_MAPPINGS as display_mappings
        )
        NODE_CLASS_MAPPINGS.update(mappings)
        NODE_DISPLAY_NAME_MAPPINGS.update(display_mappings)
        logger.info(f"[Flux MakeUp] Successfully loaded {len(mappings)} nodes via full path import")
        return True
    except ImportError as e:
        import_errors.append(f"Full path import failed: {e}")
    
    # Strategy 2: Direct import (when running from custom_nodes)
    try:
        from stable_makeup_nodes import (
            NODE_CLASS_MAPPINGS as mappings,
            NODE_DISPLAY_NAME_MAPPINGS as display_mappings
        )
        NODE_CLASS_MAPPINGS.update(mappings)
        NODE_DISPLAY_NAME_MAPPINGS.update(display_mappings)
        logger.info(f"[Flux MakeUp] Successfully loaded {len(mappings)} nodes via direct import")
        return True
    except ImportError as e:
        import_errors.append(f"Direct import failed: {e}")
    
    # Strategy 3: Try importing from ComfyUI_Stable_Makeup package
    try:
        from ComfyUI_Stable_Makeup.stable_makeup_nodes import (
            NODE_CLASS_MAPPINGS as mappings,
            NODE_DISPLAY_NAME_MAPPINGS as display_mappings
        )
        NODE_CLASS_MAPPINGS.update(mappings)
        NODE_DISPLAY_NAME_MAPPINGS.update(display_mappings)
        logger.info(f"[Flux MakeUp] Successfully loaded {len(mappings)} nodes via package import")
        return True
    except ImportError as e:
        import_errors.append(f"Package import failed: {e}")
    
    # All strategies failed - log the errors
    logger.warning("[Flux MakeUp] Failed to load node mappings. Errors encountered:")
    for error in import_errors:
        logger.warning(f"  - {error}")
    
    return False

# Attempt to load node mappings
_load_success = _load_node_mappings()

# Validate that required nodes are present
REQUIRED_NODES = ["StableMakeup_LoadModel", "StableMakeup_Sampler"]
missing_nodes = [node for node in REQUIRED_NODES if node not in NODE_CLASS_MAPPINGS]

if missing_nodes and _load_success:
    logger.warning(f"[Flux MakeUp] Missing required nodes: {missing_nodes}")
elif not _load_success:
    logger.error("[Flux MakeUp] Node loading failed. Custom nodes will not be available.")
else:
    logger.info(f"[Flux MakeUp] All {len(REQUIRED_NODES)} required nodes loaded successfully")
    logger.info(f"[Flux MakeUp] Registered nodes: {list(NODE_CLASS_MAPPINGS.keys())}")

# Web directory for any custom UI components (optional)
WEB_DIRECTORY = "./web"

# Version information
__version__ = "1.0.0"
__author__ = "Flux MakeUp Contributors"

# NODES_LIST is an alternative to NODE_CLASS_MAPPINGS used by some ComfyUI versions
# Provide both for maximum compatibility
NODES_LIST = list(NODE_CLASS_MAPPINGS.keys())

# Export the required ComfyUI variables
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS", 
    "NODES_LIST",
    "WEB_DIRECTORY",
    "__version__"
]
