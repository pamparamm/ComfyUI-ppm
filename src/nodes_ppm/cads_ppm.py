import copy
from typing import Any, Callable

import torch

import comfy.samplers
from comfy.conds import CONDRegular
from comfy.model_patcher import ModelPatcher
from comfy.model_sampling import ModelSamplingContinuousEDM
from comfy_api.latest import io

COND_KEYS = ["c_concat", "c_crossattn"]


class CADSPPM(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CADSPPM",
            display_name="CADS (PPM)",
            category="model_patches/unet",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("scale", default=0.25, min=0.0, max=10.0, step=0.001),
                io.Float.Input("start_percent", default=0.0, min=0.0, max=1.0, step=0.001),
                io.Float.Input("end_percent", default=0.4, min=0.0, max=1.0, step=0.001),
            ],
            outputs=[
                io.Model.Output(),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        model: ModelPatcher = kwargs["model"]
        scale: float = kwargs["scale"]
        start_percent: float = kwargs["start_percent"]
        end_percent: float = kwargs["end_percent"]

        m: ModelPatcher = model.clone()

        # ModelSamplingContinuousEDM is used only for type-checking purposes
        model_sampling: ModelSamplingContinuousEDM = m.get_model_object("model_sampling")  # type: ignore
        sigma_start = model_sampling.percent_to_sigma(start_percent)
        if start_percent == 0.0:
            sigma_start = model_sampling.sigma_max.item()
        sigma_end = model_sampling.percent_to_sigma(end_percent)

        prev_calc_cond_batch: Callable | None = m.model_options.get("sampler_calc_cond_batch_function")  # type: ignore

        def cads_preprocess(args):
            model = args["model"]
            conds: list[list[dict[str, Any]] | None] = args["conds"]
            x = args["input"]
            timestep: torch.Tensor = args["sigma"]
            model_options = args["model_options"]

            sigma = timestep[0].item()
            if scale > 0.0 and end_percent > start_percent and sigma <= sigma_start:
                cads_percent = (sigma_start - sigma) / (sigma_start - sigma_end)
                gamma = min(max(0.0, cads_percent), 1.0)
                conds = copy.deepcopy(conds)

                for cond in conds:
                    if cond is None:
                        continue
                    for cond_i in cond:
                        model_conds: dict[str, CONDRegular] = cond_i.get("model_conds", {})
                        for model_cond_key, model_cond in model_conds.items():
                            if model_cond_key in COND_KEYS:
                                inner_cond: torch.Tensor = model_cond.cond
                                inner_cond_noisy = cls.add_noise(inner_cond, gamma, scale)
                                model_conds[model_cond_key] = model_cond._copy_with(inner_cond_noisy)

            if prev_calc_cond_batch:
                args = {
                    "conds": conds,
                    "input": x,
                    "sigma": timestep,
                    "model": model,
                    "model_options": model_options,
                }
                return prev_calc_cond_batch(args)
            else:
                return comfy.samplers.calc_cond_batch(model, conds, x, timestep, model_options)  # type: ignore

        m.set_model_sampler_calc_cond_batch_function(cads_preprocess)
        return io.NodeOutput(m)

    @classmethod
    def add_noise(cls, y: torch.Tensor, gamma: float, noise_scale: float):
        dims = tuple(range(1, len(y.shape)))
        y_mean, y_std = y.mean(dim=dims), y.std(dim=dims)
        y_noised: torch.Tensor = gamma**0.5 * y + noise_scale * (1.0 - gamma) ** 0.5 * torch.randn_like(y)
        y_noised_norm = (y_noised - y_noised.mean(dim=dims)) / y_noised.std(dim=dims)
        return y_noised_norm * y_std + y_mean


NODES = [CADSPPM]
