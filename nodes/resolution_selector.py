"""
Resolution Selector node for Anima / 16-channel latent workflows.

Registered type: AnimaResolutionSelector (workflow compatibility)
Display name: Resolution Selector
"""

from __future__ import annotations

import re

import torch

import comfy.model_management as model_management


# Flux1 aspect patterns from ControlAltAI Megapixel Calculator
# (nodes/controlaltai/megapixel_calculator_node.py), sized at 1.0 MP / divisible_by 64.
# High-* entries are the same ratios at 1.5 MP.
_PRESETS = [
    # --- 1.0 MP (Flux1 default) ---
    "1:1 (Perfect Square) (1024x1024)",
    "2:3 (Classic Portrait) (832x1216)",
    "3:4 (Golden Ratio) (896x1152)",
    "3:5 (Elegant Vertical) (768x1280)",
    "4:5 (Artistic Frame) (896x1088)",
    "5:7 (Balanced Portrait) (832x1152)",
    "5:8 (Tall Portrait) (768x1280)",
    "7:9 (Modern Portrait) (896x1152)",
    "9:16 (Slim Vertical) (768x1344)",
    "9:19 (Tall Slim) (704x1472)",
    "9:21 (Ultra Tall) (640x1536)",
    "9:32 (Skyline) (512x1856)",
    "3:2 (Golden Landscape) (1216x832)",
    "4:3 (Classic Landscape) (1152x896)",
    "5:3 (Wide Horizon) (1280x768)",
    "5:4 (Balanced Frame) (1088x896)",
    "7:5 (Elegant Landscape) (1152x832)",
    "8:5 (Cinematic View) (1280x768)",
    "9:7 (Artful Horizon) (1152x896)",
    "16:9 (Panorama) (1344x768)",
    "19:9 (Cinematic Ultrawide) (1472x704)",
    "21:9 (Epic Ultrawide) (1536x640)",
    "32:9 (Extreme Ultrawide) (1856x512)",
    # --- 1.5 MP (high) ---
    "High 1:1 (Perfect Square) (1216x1216)",
    "High 2:3 (Classic Portrait) (1024x1536)",
    "High 3:2 (Golden Landscape) (1536x1024)",
    "High 9:16 (Slim Vertical) (896x1600)",
    "High 16:9 (Panorama) (1600x896)",
]


def _parse_wh(preset: str) -> tuple[int, int] | None:
    m = re.search(r"(\d+)\s*[xX]\s*(\d+)", preset or "")
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


class AnimaResolutionSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["Preset", "Custom"], {"default": "Preset"}),
                "preset": (_PRESETS, {"default": "4:5 (Artistic Frame) (896x1088)"}),
                "custom_width": (
                    "INT",
                    {"default": 1024, "min": 16, "max": 8192, "step": 8},
                ),
                "custom_height": (
                    "INT",
                    {"default": 1280, "min": 16, "max": 8192, "step": 8},
                ),
                "hires_scale": (
                    "FLOAT",
                    {"default": 1.3, "min": 0.1, "max": 8.0, "step": 0.05},
                ),
                "batch_size": (
                    "INT",
                    {"default": 1, "min": 1, "max": 64, "step": 1},
                ),
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "LATENT", "STRING")
    RETURN_NAMES = (
        "width",
        "height",
        "hires_width",
        "hires_height",
        "latent",
        "info",
    )
    FUNCTION = "select"
    CATEGORY = "ussoewwin/resolution"

    def select(
        self,
        mode,
        preset,
        custom_width,
        custom_height,
        hires_scale,
        batch_size,
    ):
        if mode == "Custom":
            width, height = int(custom_width), int(custom_height)
            source = "custom"
        else:
            parsed = _parse_wh(preset)
            if parsed is None:
                width, height = int(custom_width), int(custom_height)
                source = f"preset-fallback:{preset}"
            else:
                width, height = parsed
                source = f"preset:{preset}"

        hires_width = max(16, int(round(width * float(hires_scale))))
        hires_height = max(16, int(round(height * float(hires_scale))))
        hires_width -= hires_width % 8
        hires_height -= hires_height % 8

        device = model_management.intermediate_device()
        dtype = model_management.intermediate_dtype()
        latent = torch.zeros(
            [int(batch_size), 16, height // 8, width // 8],
            device=device,
            dtype=dtype,
        )
        info = (
            f"Resolution Selector mode={mode} source={source} "
            f"{width}x{height} hires={hires_width}x{hires_height} "
            f"scale={hires_scale} batch={batch_size}"
        )
        return (
            width,
            height,
            hires_width,
            hires_height,
            {"samples": latent},
            info,
        )


NODE_CLASS_MAPPINGS = {
    "AnimaResolutionSelector": AnimaResolutionSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaResolutionSelector": "Resolution Selector",
}
