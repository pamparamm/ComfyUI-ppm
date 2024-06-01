import math
import torch

import comfy.model_management
from nodes import MAX_RESOLUTION


MIN_RATIO = 0.15
MAX_RATIO = 1 / MIN_RATIO


class EmptyLatentImageAR:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "resolution": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                "ratio": ("FLOAT", {"default": 1.0, "min": MIN_RATIO, "max": MAX_RATIO, "step": 0.001}),
                "step": ("INT", {"default": 64, "min": 8, "max": 128, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"

    CATEGORY = "latent"

    def generate(self, resolution, ratio, step, batch_size=1):
        target_res = resolution * resolution

        h = math.sqrt(target_res / ratio)
        h_s = int((h // step) * step)
        height = min([h_s, h_s + step], key=lambda x: abs(h - x))

        w = height * ratio
        w_s = int((w // step) * step)
        width = min([w_s, w_s + step], key=lambda x: abs(target_res - x * height))

        width, height = min(max(width, 16), MAX_RESOLUTION), min(max(height, 16), MAX_RESOLUTION)

        latent = torch.zeros([batch_size, 4, height // 8, width // 8], device=self.device)
        return ({"samples": latent},)
