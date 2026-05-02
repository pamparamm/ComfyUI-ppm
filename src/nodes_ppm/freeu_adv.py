# modified (and partially simplified) version of https://github.com/WASasquatch/FreeU_Advanced (MIT License)
# code originally taken from: https://github.com/ChenyangSi/FreeU (under MIT License)

import logging
from typing import Any

import torch
import torch.fft as fft

from comfy.comfy_types.node_typing import IO, ComfyNodeABC, InputTypeDict
from comfy.model_patcher import ModelPatcher
from comfy.model_sampling import ModelSamplingDiscrete


def Fourier_filter(x, threshold, scale):
    # FFT
    if isinstance(x, list):
        x = x[0]
    if isinstance(x, torch.Tensor):
        x_freq = fft.fftn(x.float(), dim=(-2, -1))
        x_freq = fft.fftshift(x_freq, dim=(-2, -1))

        B, C, H, W = x_freq.shape
        mask = torch.ones((B, C, H, W), device=x.device)

        crow, ccol = H // 2, W // 2
        mask[..., crow - threshold : crow + threshold, ccol - threshold : ccol + threshold] = scale

        x_freq = x_freq * mask

        # IFFT
        x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
        x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real

        return x_filtered.to(x.dtype)

    return x


class FreeU2PPM(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "model": (IO.MODEL, {}),
                "input_block": (IO.BOOLEAN, {"default": False}),
                "middle_block": (IO.BOOLEAN, {"default": False}),
                "output_block": (IO.BOOLEAN, {"default": False}),
                "slice_b1": (IO.INT, {"default": 640, "min": 64, "max": 1280, "step": 1}),
                "slice_b2": (IO.INT, {"default": 320, "min": 64, "max": 640, "step": 1}),
                "b1": (IO.FLOAT, {"default": 1.1, "min": 0.0, "max": 10.0, "step": 0.001}),
                "b2": (IO.FLOAT, {"default": 1.2, "min": 0.0, "max": 10.0, "step": 0.001}),
                "s1": (IO.FLOAT, {"default": 0.9, "min": 0.0, "max": 10.0, "step": 0.001}),
                "s2": (IO.FLOAT, {"default": 0.2, "min": 0.0, "max": 10.0, "step": 0.001}),
                "start_percent": (IO.FLOAT, {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": False}),
                "end_percent": (IO.FLOAT, {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": False}),
            },
            "optional": {
                "threshold": (IO.INT, {"default": 1, "max": 10, "min": 1, "step": 1}),
            },
        }

    RETURN_TYPES = (IO.MODEL,)
    FUNCTION = "patch"

    CATEGORY = "_for_testing"

    def patch(
        self,
        model: ModelPatcher,
        input_block: bool,
        middle_block: bool,
        output_block: bool,
        slice_b1: int,
        slice_b2: int,
        b1: float,
        b2: float,
        s1: float,
        s2: float,
        threshold: int = 1,
        start_percent: float = 0.0,
        end_percent: float = 1.0,
    ):
        model_sampling: ModelSamplingDiscrete = model.get_model_object("model_sampling")  # type: ignore
        sigma_start = model_sampling.percent_to_sigma(start_percent)
        sigma_end = model_sampling.percent_to_sigma(end_percent)

        min_slice = 64
        max_slice_b1 = 1280
        max_slice_b2 = 640
        slice_b1 = max(min(max_slice_b1, slice_b1), min_slice)
        slice_b2 = max(min(min(slice_b1, max_slice_b2), slice_b2), min_slice)

        def _hidden_mean(h):
            hidden_mean = h.mean(1).unsqueeze(1)
            B = hidden_mean.shape[0]
            hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True)
            hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
            hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (hidden_max - hidden_min).unsqueeze(
                2
            ).unsqueeze(3)
            return hidden_mean

        def block_patch(h, transformer_options):
            sigma = transformer_options["sigmas"]
            if not (sigma_end < sigma[0] <= sigma_start):
                return h

            if h.shape[1] == 1280:
                hidden_mean = _hidden_mean(h)
                h[:, :slice_b1] = h[:, :slice_b1] * ((b1 - 1) * hidden_mean + 1)
            if h.shape[1] == 640:
                hidden_mean = _hidden_mean(h)
                h[:, :slice_b2] = h[:, :slice_b2] * ((b2 - 1) * hidden_mean + 1)
            return h

        def block_after_patch_middle(kwargs: dict[str, Any]):
            h = block_patch(kwargs["h"], kwargs["transformer_options"])
            return {"h": h}

        def block_patch_hsp(h, hsp, transformer_options):
            sigma = transformer_options["sigmas"]
            if not (sigma_end < sigma[0] <= sigma_start):
                return h, hsp

            if h.shape[1] == 1280:
                h = block_patch(h, transformer_options)
                hsp = Fourier_filter(hsp, threshold=threshold, scale=s1)
            if h.shape[1] == 640:
                h = block_patch(h, transformer_options)
                hsp = Fourier_filter(hsp, threshold=threshold, scale=s2)
            return h, hsp

        m = model.clone()
        if output_block:
            logging.debug("Patching output block")
            m.set_model_output_block_patch(block_patch_hsp)
        if input_block:
            logging.debug("Patching input block")
            m.set_model_input_block_patch(block_patch)
        if middle_block:
            logging.debug("Patching middle block")
            m.set_model_middle_block_after_patch(block_after_patch_middle)
        return (m,)


NODE_CLASS_MAPPINGS = {
    "FreeU2PPM": FreeU2PPM,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FreeU2PPM": "FreeU V2 (PPM)",
}
