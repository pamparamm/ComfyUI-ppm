import torch
from comfy.comfy_types.node_typing import IO, ComfyNodeABC, InputTypeDict
from comfy.model_base import ModelSamplingDiscrete
from comfy.model_patcher import ModelPatcher


class ConvertTimestepToSigma(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "model": (IO.MODEL, {}),
                "mode": (IO.COMBO, {"default": "percent", "options": ["none", "percent", "schedule_step"]}),
            },
            "optional": {
                "percent": (IO.FLOAT, {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "schedule_sigmas": (IO.SIGMAS, {}),
                "schedule_step": (IO.INT, {"default": 0, "min": 0, "max": 999}),
            },
        }

    RETURN_TYPES = (IO.FLOAT,)
    FUNCTION = "convert"
    CATEGORY = "sampling/custom_sampling/sigmas"

    def convert(
        self,
        model: ModelPatcher,
        mode: str,
        percent: float = 0.0,
        schedule_sigmas: list[torch.Tensor] = [],
        schedule_step: int = 0,
    ):
        model_sampling: ModelSamplingDiscrete = model.get_model_object("model_sampling")  # type: ignore
        sigma = -1.0

        if mode == "percent":
            sigma = model_sampling.percent_to_sigma(percent)
        elif mode == "schedule_step" and schedule_sigmas is not None:
            sigma = schedule_sigmas[schedule_step]

        return (sigma,)
