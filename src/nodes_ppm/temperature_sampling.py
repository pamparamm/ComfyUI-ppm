# Implementation of 2607.10853 'Diversify Diffusion with Temperature Sampling and Variance-Corrective Time Shifting'
# https://arxiv.org/abs/2607.10853
import math
from collections.abc import Callable

import torch

import comfy.patcher_extension
from comfy.model_patcher import ModelPatcher
from comfy.model_sampling import (
    ModelSamplingContinuousEDM,
    ModelSamplingDiscrete,
    ModelSamplingDiscreteFlow,
    ModelSamplingFlux,
)
from comfy_api.latest import io


class TemperatureSamplingPPM(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TemperatureSamplingPPM",
            display_name="Temperature Sampling (PPM)",
            category="model_patches/unet",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input(
                    "gamma",
                    default=0.8,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    tooltip="Inverse temperature. < 1.0 increases sampling diversity, 1.0 disables temperature sampling.",
                ),
                io.Float.Input(
                    "start_percent",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    tooltip=("Presets from original paper: early=0.0, middle=0.3333, late=0.6667."),
                ),
                io.Float.Input(
                    "end_percent",
                    default=0.3333,
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    tooltip=("Presets from original paper: early=0.3333, middle=0.6667, late=1.0."),
                ),
            ],
            outputs=[
                io.Model.Output(),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        model: ModelPatcher = kwargs["model"]
        gamma: float = kwargs["gamma"]
        start_percent: float = kwargs["start_percent"]
        end_percent: float = kwargs["end_percent"]

        m: ModelPatcher = model.clone()

        if gamma == 1.0:
            return io.NodeOutput(m)

        model_sampling = m.get_model_object("model_sampling")  # type: ignore
        if isinstance(model_sampling, (ModelSamplingDiscreteFlow, ModelSamplingFlux)):
            flow = True
        elif isinstance(model_sampling, (ModelSamplingDiscrete, ModelSamplingContinuousEDM)):
            flow = False
        else:
            raise ValueError(f"Unsupported model sampling: {type(model_sampling).__name__}")

        sigma_start = model_sampling.percent_to_sigma(start_percent)
        sigma_end = model_sampling.percent_to_sigma(end_percent)

        gamma_sqrt = math.sqrt(gamma)

        def temperature_wrapper(
            executor: Callable[..., torch.Tensor], x: torch.Tensor, sigma: torch.Tensor, *args, **kwargs
        ):
            sigma_ = sigma[0].item()

            if start_percent >= end_percent or not sigma_end <= sigma_ <= sigma_start:
                return executor(x, sigma, *args, **kwargs)

            sigma_tilde = sigma * gamma_sqrt

            if flow:
                d = 1.0 - (1.0 - gamma_sqrt) * sigma_
                return executor(x / d, sigma_tilde / d, *args, **kwargs)

            return executor(x, sigma_tilde, *args, **kwargs)

        m.add_wrapper(comfy.patcher_extension.WrappersMP.APPLY_MODEL, temperature_wrapper)
        return io.NodeOutput(m)


NODES = [TemperatureSamplingPPM]
