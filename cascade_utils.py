import math
import torch
import nodes


def remap(val, min_val, max_val, min_map, max_map):
    return (val - min_val) / (max_val - min_val) * (max_map - min_map) + min_map


def clamp(val, min, max):
    if val < min:
        return min
    elif val > max:
        return max
    else:
        return val


def compression_curve(x):
    return math.pow(x, 0.9)


def calc_compression_factor(width, height):
    min_len = min(width, height)
    max_len = max(width, height)
    # Clamp ratio to a specific length
    ratio = max_len / min_len
    ratio = clamp(ratio, 1, 2.25)
    # Remap the aspect ratio from linear to an eased curve
    r_factor = compression_curve(remap(ratio, 1, 2.25, 0, 1))
    # Figure out if the max latent length is clamped at 32 to 60
    max_fac_len = int(clamp(remap(r_factor, 0, 1, 48, 60), 32, 60))

    final_compression_factor = 0
    found_factor = False
    # Start from the highest compression factor as lower factors have better quality
    for compression in range(80, 31, -1):
        # Find our current latent edge
        latent_size = (max_len) // compression
        if latent_size <= max_fac_len:
            final_compression_factor = compression
            found_factor = True

        # Fixes extreme aspect ratios snapping to 32
        if not found_factor:
            final_compression_factor = 80
    return clamp(final_compression_factor, 32, 80)


class StableCascade_AutoCompLatent:
    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 256, "max": nodes.MAX_RESOLUTION, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 256, "max": nodes.MAX_RESOLUTION, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("stage_c", "stage_b")
    FUNCTION = "generate"

    CATEGORY = "latent/stable_cascade"

    def generate(self, width, height, batch_size=1):
        compression = calc_compression_factor(width, height)
        c_latent = torch.zeros([batch_size, 16, height // compression, width // compression])
        b_latent = torch.zeros([batch_size, 4, height // 4, width // 4])
        return ({"samples": c_latent}, {"samples": b_latent})
