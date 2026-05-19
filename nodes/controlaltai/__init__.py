# Python 3.13 Compatible
print("\n\033[32mInitializing ControlAltAI Nodes\033[0m")  # Fixed green reset

try:
    from .megapixel_calculator_node import MegapixelCalculatorNode
    from .flux_sampler_node import FluxSampler
    from .flux_union_controlnet_node import FluxUnionControlNetApply
    from .boolean_basic_node import BooleanBasic
    from .boolean_reverse_node import BooleanReverse
    from .get_image_size_ratio_node import GetImageSizeRatio
    from .integer_settings_node import IntegerSettings
    from .integer_settings_advanced_node import IntegerSettingsAdvanced
    from .perturbation_texture_node import PerturbationTexture
    from .text_bridge_node import TextBridge
    from .two_way_switch_node import TwoWaySwitch
    from .three_way_switch_node import ThreeWaySwitch
except ImportError:
    # Fallback to absolute imports for Python 3.13 compatibility
    from megapixel_calculator_node import MegapixelCalculatorNode
    from flux_sampler_node import FluxSampler
    from flux_union_controlnet_node import FluxUnionControlNetApply
    from boolean_basic_node import BooleanBasic
    from boolean_reverse_node import BooleanReverse
    from get_image_size_ratio_node import GetImageSizeRatio
    from integer_settings_node import IntegerSettings
    from integer_settings_advanced_node import IntegerSettingsAdvanced
    from perturbation_texture_node import PerturbationTexture
    from text_bridge_node import TextBridge
    from two_way_switch_node import TwoWaySwitch
    from three_way_switch_node import ThreeWaySwitch

NODE_CLASS_MAPPINGS = {
    "MegapixelCalculatorNode": MegapixelCalculatorNode,
    "FluxSampler": FluxSampler,
    "FluxUnionControlNetApply": FluxUnionControlNetApply,
    "BooleanBasic": BooleanBasic,
    "BooleanReverse": BooleanReverse,
    "GetImageSizeRatio": GetImageSizeRatio,
    "IntegerSettings": IntegerSettings,
    "IntegerSettingsAdvanced": IntegerSettingsAdvanced,
    "PerturbationTexture": PerturbationTexture,
    "TextBridge": TextBridge,
    "TwoWaySwitch": TwoWaySwitch,
    "ThreeWaySwitch": ThreeWaySwitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MegapixelCalculatorNode": "ControlAltAI: Megapixel Calculator",
    "FluxSampler": "ControlAltAI: Advanced Sampler",
    "FluxUnionControlNetApply": "ControlAltAI: Union ControlNet Apply",
    "BooleanBasic": "ControlAltAI: Boolean Basic",
    "BooleanReverse": "ControlAltAI: Boolean Reverse",
    "GetImageSizeRatio": "ControlAltAI: Get Image Size Ratio",
    "IntegerSettings": "ControlAltAI: Integer Settings",
    "IntegerSettingsAdvanced": "ControlAltAI: Integer Settings Advanced",
    "PerturbationTexture": "ControlAltAI: Perturbation Texture",
    "TextBridge": "ControlAltAI: Text Bridge",
    "TwoWaySwitch": "ControlAltAI: Switch (Two Way)",
    "ThreeWaySwitch": "ControlAltAI: Switch (Three Way)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]