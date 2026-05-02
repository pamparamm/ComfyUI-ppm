from typing import Any, Callable

import torch
import copy

from comfy.conds import CONDRegular
import comfy.samplers
from comfy.model_patcher import ModelPatcher
from comfy.model_sampling import ModelSamplingDiscrete
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

        m = model.clone()

        model_sampling: ModelSamplingDiscrete = m.get_model_object("model_sampling")  # type: ignore
        sigma_start = model_sampling.percent_to_sigma(start_percent)
        sigma_end = model_sampling.percent_to_sigma(end_percent)

        prev_calc_cond_batch: Callable | None = m.model_options.get("sampler_calc_cond_batch_function")  # type: ignore

        def cads_preprocess(args):
            model = args["model"]
            conds: list[list[dict[str, Any]] | None] = args["conds"]
            x = args["input"]
            timestep: torch.Tensor = args["sigma"]
            model_options = args["model_options"]

            sigma = timestep[0].item()
            if sigma >= sigma_end and sigma <= sigma_start:
                conds = copy.deepcopy(conds)

                for cond in conds:
                    if cond is None:
                        continue
                    for cond_i in cond:
                        model_conds: dict[str, CONDRegular] = cond_i.get("model_conds", {})
                        for model_cond_key, model_cond in model_conds.items():
                            if model_cond_key in COND_KEYS:
                                inner_cond: torch.Tensor = model_cond.cond
                                inner_cond_noisy = cls.add_noise(inner_cond, scale)
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
    def sigma_to_percent(cls, model_sampling: ModelSamplingDiscrete, sigma: torch.Tensor):
        timestep = model_sampling.timestep(sigma)[0].item()
        percent = timestep / (model_sampling.num_timesteps - 1)
        return percent

    @classmethod
    def add_noise(cls, y: torch.Tensor, noise_scale: float = 0.25, psi: float = 1.0):
        y_mean, y_std = y.mean(), y.std()
        y = (1.0 - noise_scale) * y + noise_scale * torch.randn_like(y)
        if psi != 0.0:
            y_scaled = (y - y.mean()) / y.std() * y_std + y_mean
            y = psi * y_scaled + (1 - psi) * y
        return y


NODES = [CADSPPM]
