import math
from typing import Callable, Literal

import torch

import comfy.samplers
from comfy.model_patcher import ModelPatcher
from comfy.model_sampling import ModelSamplingContinuousEDM
from comfy_api.latest import io
from comfy_extras.nodes_tcfg import score_tangential_damping


# Based on Applying Guidance in a Limited Interval Improves Sample and Distribution Quality in Diffusion Models by Kynkäänniemi et al.
class GuidanceLimiter(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Guidance Limiter",
            display_name="Guidance Limiter",
            category="model_patches/unet",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("sigma_start", default=5.42, min=-1.0, max=10000.0, step=0.01, round=False),
                io.Float.Input("sigma_end", default=0.28, min=-1.0, max=10000.0, step=0.01, round=False),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        model: ModelPatcher = kwargs["model"]
        sigma_start: float = kwargs["sigma_start"]
        sigma_end: float = kwargs["sigma_end"]

        def limited_cfg(args):
            x_cfg = args["denoised"]
            cond = args["cond_denoised"]
            sigma = args["sigma"]

            if (sigma_start >= 0 and sigma[0] > sigma_start) or (sigma_end >= 0 and sigma[0] <= sigma_end):
                return cond

            return x_cfg

        m: ModelPatcher = model.clone()
        m.set_model_sampler_post_cfg_function(limited_cfg)
        return io.NodeOutput(m)


class Guider_CFGLimiter(comfy.samplers.CFGGuider):
    def set_limits(self, sigma_start, sigma_end):
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        cfg = self.cfg
        sigma = self.inner_model.model_sampling.sigma(timestep)

        if self.sigma_start >= 0 and sigma[0] > self.sigma_start:
            cfg = 1

        if self.sigma_end >= 0 and sigma[0] <= self.sigma_end:
            cfg = 1

        return comfy.samplers.sampling_function(
            self.inner_model,
            x,
            timestep,
            self.conds.get("negative", None),
            self.conds.get("positive", None),
            cfg,
            model_options=model_options,
            seed=seed,
        )


class CFGLimiterGuider(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CFGLimiterGuider",
            display_name="CFGLimiterGuider",
            category="sampling/custom_sampling/guiders",
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Float.Input("cfg", default=8.0, min=0.0, max=100.0, step=0.1, round=0.01),
                io.Float.Input("sigma_start", default=5.42, min=-1.0, max=10000.0, step=0.01, round=False),
                io.Float.Input("sigma_end", default=0.28, min=-1.0, max=10000.0, step=0.01, round=False),
            ],
            outputs=[
                io.Guider.Output(),
            ],
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        model: ModelPatcher = kwargs["model"]
        positive = kwargs["positive"]
        negative = kwargs["negative"]
        cfg: float = kwargs["cfg"]
        sigma_start: float = kwargs["sigma_start"]
        sigma_end: float = kwargs["sigma_end"]

        guider = Guider_CFGLimiter(model)
        guider.set_conds(positive, negative)
        guider.set_cfg(cfg)
        guider.set_limits(sigma_start, sigma_end)
        return io.NodeOutput(guider)


class RescaleCFGPost(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="RescaleCFGPost",
            display_name="RescaleCFGPost",
            category="model_patches/unet",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("multiplier", default=0.7, min=0.0, max=1.0, step=0.01, round=0.01),
                io.Boolean.Input("alt_mode", default=False),
                io.Float.Input("sigma_start", default=-1.0, min=-1.0, max=10000.0, step=0.01, round=False),
                io.Float.Input("sigma_end", default=-1.0, min=-1.0, max=10000.0, step=0.01, round=False),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        model: ModelPatcher = kwargs["model"]
        multiplier: float = kwargs["multiplier"]
        alt_mode: bool = kwargs["alt_mode"]
        sigma_start: float = kwargs["sigma_start"]
        sigma_end: float = kwargs["sigma_end"]

        def rescale_cfg(args):
            x_cfg = args["denoised"]
            cond = args["cond_denoised"]
            sigma = args["sigma"]
            cfg_rescale = multiplier

            if (
                cfg_rescale == 0.0
                or (sigma_start >= 0 and sigma[0] > sigma_start)
                or (sigma_end >= 0 and sigma[0] <= sigma_end)
            ):
                return x_cfg

            latent_dim = tuple(range(1, len(cond.shape)))
            ro_pos = torch.std(cond, dim=latent_dim, keepdim=True)
            ro_cfg = torch.std(x_cfg, dim=latent_dim, keepdim=True)

            x_rescaled = x_cfg * (ro_pos / ro_cfg)
            # Alternative version of RescaleCFG by madman404
            if alt_mode:
                cfg_rescale = cfg_rescale * (1 - (ro_pos / ro_cfg) ** 2)
            x_final = cfg_rescale * x_rescaled + (1.0 - cfg_rescale) * x_cfg
            return x_final

        m: ModelPatcher = model.clone()
        m.set_model_sampler_post_cfg_function(rescale_cfg)
        return io.NodeOutput(m)


# Shamelessly taken from sd-dynamic-thresholding by Alex "mcmonkey" Goodwin licensed under MIT License
class DynThresh:
    @classmethod
    def dynthresh(
        cls,
        x_cfg: torch.Tensor,
        x_mim: torch.Tensor,
        threshold_percentile: float = 1.0,
        sep_feat_channels: bool = False,
        startpoint: Literal["MEAN", "ZERO"] = "MEAN",
        variability: Literal["AD", "STD"] = "AD",
        interpolate_phi: float = 1.0,
    ):
        # Now recenter the values relative to their average rather than absolute, to allow scaling from average
        mim_flattened = x_mim.flatten(2)
        cfg_flattened = x_cfg.flatten(2)
        mim_means = mim_flattened.mean(dim=2).unsqueeze(2)
        cfg_means = cfg_flattened.mean(dim=2).unsqueeze(2)
        mim_centered = mim_flattened - mim_means
        cfg_centered = cfg_flattened - cfg_means

        if sep_feat_channels:
            if variability == "AD":
                mim_scaleref = mim_centered.abs().max(dim=2).values.unsqueeze(2)
                cfg_scaleref = torch.quantile(cfg_centered.abs(), threshold_percentile, dim=2).unsqueeze(2)
            elif variability == "STD":
                mim_scaleref = mim_centered.std(dim=2).unsqueeze(2)
                cfg_scaleref = cfg_centered.std(dim=2).unsqueeze(2)
        else:
            if variability == "AD":
                mim_scaleref = mim_centered.abs().max()
                cfg_scaleref = torch.quantile(cfg_centered.abs(), threshold_percentile)
            elif variability == "STD":
                mim_scaleref = mim_centered.std()
                cfg_scaleref = cfg_centered.std()

        if startpoint == "MEAN":
            if variability == "AD":
                # Get the maximum value of all datapoints (with an optional threshold percentile on the uncond)
                max_scaleref = torch.maximum(mim_scaleref, cfg_scaleref)
                # Clamp to the max
                cfg_clamped = cfg_centered.clamp(-max_scaleref, max_scaleref)
                # Now shrink from the max to normalize and grow to the mimic scale (instead of the CFG scale)
                cfg_renormalized = (cfg_clamped / max_scaleref) * mim_scaleref
            elif variability == "STD":
                cfg_renormalized = (cfg_centered / cfg_scaleref) * mim_scaleref
            # Now add it back onto the averages to get into real scale again and return
            result = cfg_renormalized + cfg_means
        elif startpoint == "ZERO":
            scaling_factor = mim_scaleref / cfg_scaleref
            result = cfg_flattened * scaling_factor

        result = result.unflatten(2, x_cfg.shape[2:])

        if interpolate_phi != 1.0:
            result = result * interpolate_phi + x_cfg * (1.0 - interpolate_phi)

        return result


class DynamicThresholdingSimplePost(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DynamicThresholdingSimplePost",
            display_name="DynamicThresholdingSimplePost",
            category="model_patches/unet",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("mimic_scale", default=8.0, min=0.0, max=100.0, step=0.1, round=0.01),
                io.Float.Input("threshold_percentile", default=1.0, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        model: ModelPatcher = kwargs["model"]
        mimic_scale: float = kwargs["mimic_scale"]
        threshold_percentile: float = kwargs["threshold_percentile"]
        return DynamicThresholdingPost.execute(
            model=model,
            mimic_scale=mimic_scale,
            threshold_percentile=threshold_percentile,
            separate_feature_channels=False,
            scaling_startpoint="MEAN",
            variability_measure="AD",
            interpolate_phi=1.0,
        )


class DynamicThresholdingPost(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DynamicThresholdingPost",
            display_name="DynamicThresholdingFullPost",
            category="model_patches/unet",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("mimic_scale", default=8.0, min=0.0, max=100.0, step=0.1, round=0.01),
                io.Float.Input("threshold_percentile", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Boolean.Input("separate_feature_channels", default=False),
                io.Combo.Input("scaling_startpoint", default="MEAN", options=["MEAN", "ZERO"]),
                io.Combo.Input("variability_measure", default="AD", options=["AD", "STD"]),
                io.Float.Input("interpolate_phi", default=1.0, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        model: ModelPatcher = kwargs["model"]
        mimic_scale: float = kwargs["mimic_scale"]
        threshold_percentile: float = kwargs["threshold_percentile"]
        separate_feature_channels: bool = kwargs["separate_feature_channels"]
        scaling_startpoint: Literal["MEAN", "ZERO"] = kwargs["scaling_startpoint"]
        variability_measure: Literal["AD", "STD"] = kwargs["variability_measure"]
        interpolate_phi: float = kwargs["interpolate_phi"]

        def dynthresh_cfg(args):
            x_cfg = args["denoised"]
            cond = args["cond_denoised"]
            uncond = args["uncond_denoised"]
            cond_scale = args["cond_scale"]

            if mimic_scale == cond_scale:
                return x_cfg

            x_mim = uncond + (cond - uncond) * mimic_scale

            return DynThresh.dynthresh(
                x_cfg,
                x_mim,
                threshold_percentile,
                separate_feature_channels,
                scaling_startpoint,
                variability_measure,
                interpolate_phi,
            )

        m: ModelPatcher = model.clone()
        m.set_model_sampler_post_cfg_function(dynthresh_cfg)
        return io.NodeOutput(m)


class RenormCFGPost(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="RenormCFGPost",
            display_name="RenormCFGPost",
            category="model_patches/unet",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("renorm_cfg", default=1.0, min=0.0, max=100.0, step=0.01),
                io.Float.Input("sigma_start", default=-1.0, min=-1.0, max=10000.0, step=0.01, round=False),
                io.Float.Input("sigma_end", default=-1.0, min=-1.0, max=10000.0, step=0.01, round=False),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        model: ModelPatcher = kwargs["model"]
        renorm_cfg: float = kwargs["renorm_cfg"]
        sigma_start: float = kwargs["sigma_start"]
        sigma_end: float = kwargs["sigma_end"]

        def renorm_cfg_func(args):
            x_cfg = args["denoised"]
            cond = args["cond_denoised"]
            sigma = args["sigma"]

            if (
                renorm_cfg == 0.0
                or (sigma_start >= 0 and sigma[0] > sigma_start)
                or (sigma_end >= 0 and sigma[0] <= sigma_end)
            ):
                return x_cfg

            latent_dim = tuple(range(1, len(cond.shape)))
            cond_norm = torch.linalg.vector_norm(cond, dim=latent_dim, keepdim=True) * renorm_cfg
            cfg_norm = torch.linalg.vector_norm(x_cfg, dim=latent_dim, keepdim=True)
            return x_cfg * (cond_norm / cfg_norm).clamp(max=1.0)

        m: ModelPatcher = model.clone()
        m.set_model_sampler_post_cfg_function(renorm_cfg_func)
        return io.NodeOutput(m)


# Slightly modified version of TCFG from ComfyUI
class TCFGAdvanced(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TCFGAdvanced",
            display_name="Tangential Damping CFG (Advanced)",
            category="advanced/guidance",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("multiplier", default=0.5, min=0.0, max=1.0, step=0.01, round=0.01),
                io.Float.Input("sigma_start", default=-1.0, min=-1.0, max=10000.0, step=0.01, round=False),
                io.Float.Input("sigma_end", default=-1.0, min=-1.0, max=10000.0, step=0.01, round=False),
            ],
            outputs=[
                io.Model.Output(),
            ],
            description="TCFG – Tangential Damping CFG (2503.18137)\n\nRefine the uncond (negative) to align with the cond (positive) for improving quality.",
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        model: ModelPatcher = kwargs["model"]
        multiplier: float = kwargs["multiplier"]
        sigma_start: float = kwargs["sigma_start"]
        sigma_end: float = kwargs["sigma_end"]

        m: ModelPatcher = model.clone()

        def tangential_damping_cfg(args):
            #  Assume [cond, uncond, ...]
            x: torch.Tensor = args["input"]
            conds_out: list[torch.Tensor] = args["conds_out"]
            sigma: torch.Tensor = args["sigma"]
            if (
                len(conds_out) <= 1
                or None in args["conds"][:2]
                or (sigma_start >= 0 and sigma[0] > sigma_start)
                or (sigma_end >= 0 and sigma[0] <= sigma_end)
            ):
                # Skip when either cond or uncond is None
                return conds_out
            cond_pred = conds_out[0]
            uncond_pred = conds_out[1]
            uncond_td = score_tangential_damping(x - cond_pred, x - uncond_pred)
            uncond_pred_td = x - uncond_td
            uncond_pred_final = multiplier * uncond_pred_td + (1.0 - multiplier) * uncond_pred
            return [cond_pred, uncond_pred_final] + conds_out[2:]

        m.set_model_sampler_pre_cfg_function(tangential_damping_cfg)
        return io.NodeOutput(m)


class SkipFirstStepCFG(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SkipFirstStepCFG",
            display_name="Skip CFG on first step",
            category="model_patches/unet",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("skip_percent", default=0.01, min=0.0, max=1.0, step=0.001),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        model: ModelPatcher = kwargs["model"]
        skip_percent: float = kwargs["skip_percent"]

        m: ModelPatcher = model.clone()
        prev_calc_cond_batch: Callable | None = m.model_options.get("sampler_calc_cond_batch_function")  # type: ignore

        def skip_first_step_cfg(args):
            model = args["model"]
            conds = args["conds"].copy()
            x = args["input"]
            timestep: torch.Tensor = args["sigma"]
            model_options = args["model_options"]

            model_sampling: ModelSamplingContinuousEDM = model.model_sampling
            sigma_val = model_sampling.percent_to_sigma(skip_percent)

            if timestep[0] > sigma_val:
                conds[1] = None

            if prev_calc_cond_batch:
                args = {"conds": conds, "input": x, "sigma": timestep, "model": model, "model_options": model_options}
                return prev_calc_cond_batch(args)
            else:
                return comfy.samplers.calc_cond_batch(model, conds, x, timestep, model_options)

        m.set_model_sampler_calc_cond_batch_function(skip_first_step_cfg)
        return io.NodeOutput(m)


class C2FG(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="C2FG",
            display_name="C2FG",
            category="model_patches/unet",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("growth_rate", default=1.0, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        model: ModelPatcher = kwargs["model"]
        growth_rate: float = kwargs["growth_rate"]

        m: ModelPatcher = model.clone()
        prev_cfg = m.model_options.get("sampler_cfg_function")
        model_sampling: ModelSamplingContinuousEDM = m.get_model_object("model_sampling")  # type: ignore
        t_min = model_sampling.timestep(model_sampling.sigma_min).item()
        t_max = model_sampling.timestep(model_sampling.sigma_max).item()

        def c2fg_cfg(args):
            scale: float = args["cond_scale"]

            timestep = model_sampling.timestep(args["sigma"])
            t = (timestep[0].item() - t_min) / (t_max - t_min)

            scale_t = scale * math.exp(growth_rate * (1.0 - t))
            if prev_cfg is not None:
                return prev_cfg({**args, "cond_scale": scale_t})

            cond = args["cond"]
            uncond = args["uncond"]
            return uncond + scale_t * (cond - uncond)

        m.set_model_sampler_cfg_function(c2fg_cfg)
        return io.NodeOutput(m)


NODES = [
    GuidanceLimiter,
    CFGLimiterGuider,
    RescaleCFGPost,
    DynamicThresholdingSimplePost,
    DynamicThresholdingPost,
    RenormCFGPost,
    TCFGAdvanced,
    SkipFirstStepCFG,
    C2FG,
]
