from comfy.model_patcher import ModelPatcher


class ConvertTimestepToSigma:
    MODES = ["none", "percent", "schedule_step"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "mode": (s.MODES, {"default": "percent"}),
            },
            "optional": {
                "percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "schedule_sigmas": ("SIGMAS",),
                "schedule_step": ("INT", {"default": 0, "min": 0, "max": 999}),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "convert"
    CATEGORY = "sampling/custom_sampling/sigmas"

    def convert(self, model: ModelPatcher, mode: str, percent: float = 0.0, schedule_sigmas=None, schedule_step: int = 0):
        model_sampling = model.get_model_object("model_sampling")
        sigma = -1.0

        if mode == "percent":
            sigma = model_sampling.percent_to_sigma(percent)
        elif mode == "schedule_step":
            sigma = schedule_sigmas[schedule_step]

        return (sigma,)
