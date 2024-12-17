from tqdm.auto import trange
import torch

from comfy.k_diffusion.sampling import to_d
import comfy.model_patcher


SAMPLER_NAMES = [
    "euler_gamma",
    "dpmpp_2m_gamma",
]


@torch.no_grad()
def sample_euler_gamma(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    cfg_pp=False,
    s_sigma_diff=2.0,
    s_sigma_max=None,
    **kwargs,
):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    uncond_denoised = None

    def post_cfg_function(args):
        nonlocal uncond_denoised
        uncond_denoised = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(
        model_options, post_cfg_function, disable_cfg1_optimization=True
    )

    sigma_max = s_sigma_max if s_sigma_max is not None else sigmas[0]

    for i in trange(len(sigmas) - 1, disable=disable):
        sigma_hat = sigmas[i]
        sigma_eps = sigmas[i] + s_sigma_diff * (sigmas[i] / sigma_max)
        if sigmas[i + 1] > 0 and sigma_eps <= sigma_max:
            sigma_hat = sigma_eps
            x = x - torch.randn_like(x) * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5

        denoised = model(x, sigma_hat * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})
        # Euler method
        if cfg_pp:
            d = to_d(x, sigma_hat, uncond_denoised)
            x = denoised + d * sigmas[i + 1]
        else:
            d = to_d(x, sigma_hat, denoised)
            dt = sigmas[i + 1] - sigma_hat
            x = x + d * dt
    return x


@torch.no_grad()
def sample_dpmpp_2m_gamma(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    cfg_pp=False,
    s_sigma_diff=2.0,
    s_sigma_max=None,
    **kwargs,
):
    """DPM-Solver++(2M)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()

    old_denoised = None
    uncond_denoised = None
    h_last = None
    h = None

    def post_cfg_function(args):
        nonlocal uncond_denoised
        uncond_denoised = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(
        model_options, post_cfg_function, disable_cfg1_optimization=True
    )

    sigma_max = s_sigma_max if s_sigma_max is not None else sigmas[0]

    for i in trange(len(sigmas) - 1, disable=disable):
        sigma_hat = sigmas[i]
        sigma_eps = sigmas[i] + s_sigma_diff * (sigmas[i] / sigma_max)
        if sigmas[i + 1] > 0 and sigma_eps <= sigma_max:
            sigma_hat = sigma_eps
            x = x - torch.randn_like(x) * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5

        denoised = model(x, sigma_hat * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})
        t, t_next = t_fn(sigma_hat), t_fn(sigmas[i + 1])
        h = t_next - t
        if cfg_pp:
            if old_denoised is None or sigmas[i + 1] == 0:
                denoised_mix = -torch.exp(-h) * uncond_denoised
            else:
                r = h_last / h
                denoised_mix = -torch.exp(-h) * uncond_denoised - torch.expm1(-h) * (1 / (2 * r)) * (denoised - old_denoised)
            x = denoised + denoised_mix + torch.exp(-h) * x
            old_denoised = uncond_denoised
            h_last = h
        else:
            if old_denoised is None or sigmas[i + 1] == 0:
                x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
            else:
                h_last = t - t_fn(sigmas[i - 1])
                r = h_last / h
                denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
                x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
            old_denoised = denoised
    return x
