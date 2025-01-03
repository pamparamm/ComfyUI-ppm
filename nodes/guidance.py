import torch
import comfy.samplers


# Based on Applying Guidance in a Limited Interval Improves Sample and Distribution Quality in Diffusion Models by Kynkäänniemi et al.
class GuidanceLimiter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "sigma_start": ("FLOAT", {"default": 5.42, "min": -1.0, "max": 10000.0, "step": 0.01, "round": False}),
                "sigma_end": ("FLOAT", {"default": 0.28, "min": -1.0, "max": 10000.0, "step": 0.01, "round": False}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/unet"

    def patch(self, model, sigma_start: float, sigma_end: float):
        def limited_cfg(args):
            x_cfg = args["denoised"]
            cond = args["cond_denoised"]
            sigma = args["sigma"]

            if (sigma_start >= 0 and sigma[0] > sigma_start) or (sigma_end >= 0 and sigma[0] <= sigma_end):
                return cond

            return x_cfg

        m = model.clone()
        m.set_model_sampler_post_cfg_function(limited_cfg)
        return (m,)


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


class CFGLimiterGuider:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sigma_start": ("FLOAT", {"default": 5.42, "min": -1.0, "max": 10000.0, "step": 0.01, "round": False}),
                "sigma_end": ("FLOAT", {"default": 0.28, "min": -1.0, "max": 10000.0, "step": 0.01, "round": False}),
            }
        }

    RETURN_TYPES = ("GUIDER",)

    FUNCTION = "get_guider"
    CATEGORY = "sampling/custom_sampling/guiders"

    def get_guider(self, model, positive, negative, cfg: float, sigma_start: float, sigma_end: float):
        guider = Guider_CFGLimiter(model)
        guider.set_conds(positive, negative)
        guider.set_cfg(cfg)
        guider.set_limits(sigma_start, sigma_end)
        return (guider,)


class RescaleCFGPost:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "cfg_rescale": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),
                "sigma_start": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 10000.0, "step": 0.01, "round": False}),
                "sigma_end": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 10000.0, "step": 0.01, "round": False}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/unet"

    def patch(self, model, cfg_rescale: float, sigma_start: float, sigma_end: float):
        def rescale_cfg(args):
            x_cfg = args["denoised"]
            cond = args["cond_denoised"]
            sigma = args["sigma"]

            if (
                cfg_rescale == 0.0
                or (sigma_start >= 0 and sigma[0] > sigma_start)
                or (sigma_end >= 0 and sigma[0] <= sigma_end)
            ):
                return x_cfg

            ro_pos = torch.std(cond, dim=(1, 2, 3), keepdim=True)
            ro_cfg = torch.std(x_cfg, dim=(1, 2, 3), keepdim=True)

            x_rescaled = x_cfg * (ro_pos / ro_cfg)
            x_final = cfg_rescale * x_rescaled + (1.0 - cfg_rescale) * x_cfg
            return x_final

        m = model.clone()
        m.set_model_sampler_post_cfg_function(rescale_cfg)
        return (m,)


# Shamelessly taken from sd-dynamic-thresholding by Alex "mcmonkey" Goodwin licensed under MIT License
def dynthresh(
    x_cfg: torch.Tensor,
    x_mim: torch.Tensor,
    threshold_percentile: float = 1.0,
    sep_feat_channels: bool = False,
    interpolate_phi: float = 1.0,
):
    ### Now recenter the values relative to their average rather than absolute, to allow scaling from average
    mim_flattened = x_mim.flatten(2)
    cfg_flattened = x_cfg.flatten(2)
    mim_means = mim_flattened.mean(dim=2).unsqueeze(2)
    cfg_means = cfg_flattened.mean(dim=2).unsqueeze(2)
    mim_centered = mim_flattened - mim_means
    cfg_centered = cfg_flattened - cfg_means

    # MEAN+AD
    if sep_feat_channels:
        mim_scaleref = mim_centered.abs().max(dim=2).values.unsqueeze(2)
        cfg_scaleref = torch.quantile(cfg_centered.abs(), threshold_percentile, dim=2).unsqueeze(2)
    else:
        mim_scaleref = mim_centered.abs().max()
        cfg_scaleref = torch.quantile(cfg_centered.abs(), threshold_percentile)

    ### Get the maximum value of all datapoints (with an optional threshold percentile on the uncond)
    max_scaleref = torch.maximum(mim_scaleref, cfg_scaleref)
    ### Clamp to the max
    cfg_clamped = cfg_centered.clamp(-max_scaleref, max_scaleref)
    ### Now shrink from the max to normalize and grow to the mimic scale (instead of the CFG scale)
    cfg_renormalized = (cfg_clamped / max_scaleref) * mim_scaleref

    ### Now add it back onto the averages to get into real scale again and return
    result = cfg_renormalized + cfg_means

    result = result.unflatten(2, x_cfg.shape[2:])

    if interpolate_phi != 1.0:
        result = result * interpolate_phi + x_cfg * (1.0 - interpolate_phi)

    return result


class DynamicThresholdingSimplePost:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "mimic_scale": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "threshold_percentile": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/unet"

    def patch(self, model, mimic_scale: float, threshold_percentile: float):
        def dynthresh_cfg(args):
            x_cfg = args["denoised"]
            cond = args["cond_denoised"]
            uncond = args["uncond_denoised"]
            cond_scale = args["cond_scale"]

            if mimic_scale == cond_scale:
                return x_cfg

            x_mim = uncond + (cond - uncond) * mimic_scale

            return dynthresh(x_cfg, x_mim, threshold_percentile, False, 1.0)

        m = model.clone()
        m.set_model_sampler_post_cfg_function(dynthresh_cfg)
        return (m,)
