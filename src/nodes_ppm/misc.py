from enum import Enum
from typing import TypedDict
import torch

from comfy.model_base import BaseModel
from comfy.model_patcher import ModelPatcher
from comfy.model_sampling import ModelSamplingDiscrete
from comfy_api.latest import io


class ConvertTimestepToSigma(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ConvertTimestepToSigma",
            display_name="Convert Timestep To Sigma",
            category="sampling/custom_sampling/sigmas",
            inputs=[
                io.Model.Input("model"),
                io.DynamicCombo.Input(
                    "mode",
                    options=[
                        io.DynamicCombo.Option(
                            cls.ModeType.PERCENT,
                            [
                                io.Float.Input("percent", default=0.0, min=0.0, max=1.0, step=0.0001),
                                io.Boolean.Input(
                                    "return_actual_sigma",
                                    default=False,
                                    tooltip="Return the actual sigma value instead of the value used for interval checks.\nThis only affects results at 0.0 and 1.0.",
                                ),
                            ],
                        ),
                        io.DynamicCombo.Option(
                            cls.ModeType.SCHEDULE_STEP,
                            [
                                io.Sigmas.Input("schedule_sigmas"),
                                io.Int.Input("schedule_step", default=0, min=0, max=999),
                            ],
                        ),
                    ],
                ),
            ],
            outputs=[
                io.Float.Output(),
            ],
        )

    CONVERT_MODES = ["percent", "schedule_step"]

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        model: ModelPatcher = kwargs["model"]
        mode: dict = kwargs["mode"]
        selected_mode = mode["mode"]

        model_sampling: ModelSamplingDiscrete = model.get_model_object("model_sampling")  # type: ignore
        sigma = -1.0

        if selected_mode == "percent":
            percent: float = mode["percent"]
            return_actual_sigma: bool = mode["return_actual_sigma"]
            sigma = model_sampling.percent_to_sigma(percent)
            if return_actual_sigma:
                if percent == 0.0:
                    sigma = model_sampling.sigma_max.item()
                elif percent == 1.0:
                    sigma = model_sampling.sigma_min.item()
        elif selected_mode == "schedule_step":
            schedule_sigmas: list[torch.Tensor] = mode["schedule_sigmas"]
            schedule_step: int = mode["schedule_step"]
            sigma = schedule_sigmas[schedule_step]

        return io.NodeOutput(sigma)

    class ModeType(str, Enum):
        PERCENT = "percent"
        SCHEDULE_STEP = "schedule_step"


class EpsilonScalingPPM(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EpsilonScalingPPM",
            display_name="Epsilon Scaling (PPM)",
            category="model_patches/unet",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input(
                    "scaling_factor",
                    default=1.005,
                    min=0.5,
                    max=1.5,
                    step=0.001,
                    display_mode=io.NumberDisplay.number,
                ),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        model: ModelPatcher = kwargs["model"]
        scaling_factor: float = kwargs["scaling_factor"]

        if scaling_factor == 0:
            scaling_factor = 1e-9

        def epsilon_scaling_function(args):
            model: BaseModel = args["model"]
            x_cfg = args["denoised"]
            x = args["input"]
            sigma = args["sigma"]

            model_sampling: ModelSamplingDiscrete = model.model_sampling
            zsnr = getattr(model_sampling, "zsnr", False)

            if zsnr and sigma >= model_sampling.sigma_max:
                return x_cfg

            noise_pred = x - x_cfg
            scaled_noise_pred = noise_pred / scaling_factor
            new_denoised = x - scaled_noise_pred

            return new_denoised

        m = model.clone()
        m.set_model_sampler_post_cfg_function(epsilon_scaling_function)
        return io.NodeOutput(m)


NODES = [ConvertTimestepToSigma, EpsilonScalingPPM]
