"""
ComfyUI-NunchakuFluxLoraStacker

A standalone ComfyUI custom node for Nunchaku FLUX LoRA Stacking.
"""

import torch
import sys
import types

if not hasattr(torch, 'distributed') or not hasattr(torch.distributed, 'is_available') or not torch.distributed.is_available():
    dist_stub = types.ModuleType('torch.distributed')
    dist_stub.is_available = lambda: False
    dist_stub.is_initialized = lambda: False
    dist_stub.DTensor = type(None)
    dist_stub.tensor = torch.tensor
    dist_stub.get_rank = lambda: 0
    dist_stub.get_world_size = lambda: 1
    sys.modules['torch.distributed'] = dist_stub
    sys.modules['torch.distributed.tensor'] = dist_stub
    sys.modules['torch.distributed.device_mesh'] = dist_stub
    torch.distributed = dist_stub
else:
    import torch.distributed as dist
    if not hasattr(dist, 'DTensor'):
        dist.DTensor = type(None)
    if not hasattr(dist, 'tensor'):
        dist.tensor = torch.tensor

import logging

# Version information
__version__ = "1.13.0"

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Import nodes — nunchaku-dependent imports are guarded for AMD/non-NVIDIA systems
try:
    from .nodes.lora.flux import NunchakuFluxLoraStack
    from .nodes.lora.flux_v2 import GENERATED_NODES as FLUX_NODES, GENERATED_DISPLAY_NAMES as FLUX_NAMES
    _FLUX_AVAILABLE = True
except (ImportError, Exception) as e:
    logger.warning(f"[AMD patch] Skipping nunchaku FLUX nodes: {e}")
    NunchakuFluxLoraStack = None
    FLUX_NODES = {}
    FLUX_NAMES = {}
    _FLUX_AVAILABLE = False

try:
    from .nodes.lora.standard import GENERATED_NODES as STANDARD_LORA_NODES, GENERATED_DISPLAY_NAMES as STANDARD_LORA_NAMES
except (ImportError, Exception) as e:
    logger.warning(f"[AMD patch] Skipping standard LoRA nodes: {e}")
    STANDARD_LORA_NODES = {}
    STANDARD_LORA_NAMES = {}

from .nodes.lora.sdnq import GENERATED_NODES as SDNQ_LORA_NODES, GENERATED_DISPLAY_NAMES as SDNQ_LORA_NAMES
from .nodes.misc_v2 import NODE_CLASS_MAPPINGS as MISC_NODES, NODE_DISPLAY_NAME_MAPPINGS as MISC_NAMES
from .nodes.load_image_ussoewwin import NODE_CLASS_MAPPINGS as LOAD_IMAGE_NODES, NODE_DISPLAY_NAME_MAPPINGS as LOAD_IMAGE_NAMES
from .nodes.lora_analyzer_node import NODE_CLASS_MAPPINGS as ANALYZER_NODES, NODE_DISPLAY_NAME_MAPPINGS as ANALYZER_NAMES

# Add version to classes
if NunchakuFluxLoraStack is not None:
    NunchakuFluxLoraStack.__version__ = __version__
for node_class in FLUX_NODES.values():
    node_class.__version__ = __version__
for node_class in STANDARD_LORA_NODES.values():
    node_class.__version__ = __version__
for node_class in SDNQ_LORA_NODES.values():
    node_class.__version__ = __version__

# Node mappings
NODE_CLASS_MAPPINGS = {
    **({} if NunchakuFluxLoraStack is None else {"FluxLoraMultiLoader": NunchakuFluxLoraStack}),
    **FLUX_NODES,
    **STANDARD_LORA_NODES,
    **SDNQ_LORA_NODES,
    **MISC_NODES,
    **LOAD_IMAGE_NODES,
    **ANALYZER_NODES,
}

# Display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    **({} if NunchakuFluxLoraStack is None else {"FluxLoraMultiLoader": "FLUX LoRA Multi Loader (Legacy - Do Not Use in V2)"}),
    **FLUX_NAMES,
    **STANDARD_LORA_NAMES,
    **SDNQ_LORA_NAMES,
    **MISC_NAMES,
    **LOAD_IMAGE_NAMES,
    **ANALYZER_NAMES,
}

# Register JavaScript extensions
# Serve JS from ./js (used by this extension's frontend widgets)
WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__"]

logger.info(f"ComfyUI-NunchakuFluxLoraStacker: Loaded {len(NODE_CLASS_MAPPINGS)} nodes")
