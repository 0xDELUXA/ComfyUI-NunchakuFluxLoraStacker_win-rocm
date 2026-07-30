"""
Resolution Selector node for Anima / 16-channel latent workflows.

Registered type: AnimaResolutionSelector (workflow compatibility)
Display name: Resolution Selector
"""

from __future__ import annotations

import re

import torch

import comfy.model_management as model_management


_PRESETS = [
    "Portrait 4:5 (1024x1280)",
    "Square 1:1 (1024x1024)",
    "Portrait 3:4 (896x1152)",
    "Portrait 2:3 (832x1248)",
    "Portrait 9:16 (720x1280)",
    "Landscape 4:3 (1152x864)",
    "Landscape 3:2 (1248x832)",
    "Landscape 16:9 (1280x720)",
    "High Portrait 2:3 (1024x1536)",
    "High Landscape 3:2 (1536x1024)",
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
                "preset": (_PRESETS, {"default": "Portrait 4:5 (1024x1280)"}),
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
