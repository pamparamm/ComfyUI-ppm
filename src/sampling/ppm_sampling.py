from functools import partial

import torch
from tqdm.auto import trange

import comfy.model_patcher
from comfy.k_diffusion.sa_solver import get_tau_interval_func
from comfy.k_diffusion.sampling import (
    default_noise_sampler,
    ei_h_phi_1,
    ei_h_phi_2,
    half_log_snr_to_sigma,
    offset_first_sigma_for_snr,
    sigma_to_half_log_snr,
    to_d,
)

SAMPLER_NAMES: list = [
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
                denoised_mix = -torch.exp(-h) * uncond_denoised - torch.expm1(-h) * (1 / (2 * r)) * (
                    denoised - old_denoised
                )
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


@torch.no_grad()
def sample_seeds_2_scheduled(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
    r=0.5,
    solver_type="phi_1",
    tau_func=None,
):
    """SEEDS-2 - Stochastic Explicit Exponential Derivative-free Solvers (VP Data Prediction) stage 2.
    arXiv: https://arxiv.org/abs/2305.14267 (NeurIPS 2023)
    """
    if solver_type not in {"phi_1", "phi_2"}:
        raise ValueError("solver_type must be 'phi_1' or 'phi_2'")

    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    model_sampling = model.inner_model.model_patcher.get_model_object("model_sampling")
    s_noise = s_noise * getattr(model_sampling, "noise_scale", 1.0)
    sigma_fn = partial(half_log_snr_to_sigma, model_sampling=model_sampling)
    lambda_fn = partial(sigma_to_half_log_snr, model_sampling=model_sampling)
    sigmas = offset_first_sigma_for_snr(sigmas, model_sampling)

    if tau_func is None:
        # Use default interval for stochastic sampling
        start_sigma = model_sampling.percent_to_sigma(0.2)
        end_sigma = model_sampling.percent_to_sigma(0.8)
        tau_func = get_tau_interval_func(start_sigma, end_sigma, eta=eta)

    fac = 1 / (2 * r)

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})

        if sigmas[i + 1] == 0:
            x = denoised
            continue

        # sa_solver fragment
        tau_t = tau_func(sigmas[i + 1])
        inject_noise = tau_t > 0 and s_noise > 0

        lambda_s, lambda_t = lambda_fn(sigmas[i]), lambda_fn(sigmas[i + 1])
        h = lambda_t - lambda_s
        h_eta = h * (tau_t + 1)
        lambda_s_1 = torch.lerp(lambda_s, lambda_t, r)
        sigma_s_1 = sigma_fn(lambda_s_1)

        alpha_s_1 = sigma_s_1 * lambda_s_1.exp()
        alpha_t = sigmas[i + 1] * lambda_t.exp()

        # Step 1
        x_2 = sigma_s_1 / sigmas[i] * (-r * h * tau_t).exp() * x - alpha_s_1 * ei_h_phi_1(-r * h_eta) * denoised
        if inject_noise:
            sde_noise = (-2 * r * h * tau_t).expm1().neg().sqrt() * noise_sampler(sigmas[i], sigma_s_1)
            x_2 = x_2 + sde_noise * sigma_s_1 * s_noise
        denoised_2 = model(x_2, sigma_s_1 * s_in, **extra_args)

        # Step 2
        if solver_type == "phi_1":
            denoised_d = torch.lerp(denoised, denoised_2, fac)
            x = sigmas[i + 1] / sigmas[i] * (-h * tau_t).exp() * x - alpha_t * ei_h_phi_1(-h_eta) * denoised_d
        elif solver_type == "phi_2":
            b2 = ei_h_phi_2(-h_eta) / r
            b1 = ei_h_phi_1(-h_eta) - b2
            x = sigmas[i + 1] / sigmas[i] * (-h * tau_t).exp() * x - alpha_t * (b1 * denoised + b2 * denoised_2)

        if inject_noise:
            segment_factor = (r - 1) * h * tau_t
            sde_noise = sde_noise * segment_factor.exp()
            sde_noise = sde_noise + segment_factor.mul(2).expm1().neg().sqrt() * noise_sampler(sigma_s_1, sigmas[i + 1])
            x = x + sde_noise * sigmas[i + 1] * s_noise
    return x


@torch.no_grad()
def sample_er_sde_scheduled(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    s_noise=1.0,
    noise_sampler=None,
    noise_scaler=None,
    max_stage=3,
    tau_func=None,
):
    """Extended Reverse-Time SDE solver (VP ER-SDE-Solver-3). arXiv: https://arxiv.org/abs/2309.06169.
    Code reference: https://github.com/QinpengCui/ER-SDE-Solver/blob/main/er_sde_solver.py.
    """
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    model_sampling = model.inner_model.model_patcher.get_model_object("model_sampling")
    s_noise = s_noise * getattr(model_sampling, "noise_scale", 1.0)

    if tau_func is None:
        # Use default interval for stochastic sampling
        start_sigma = model_sampling.percent_to_sigma(0.2)
        end_sigma = model_sampling.percent_to_sigma(0.8)
        tau_func = get_tau_interval_func(start_sigma, end_sigma)

    def default_er_sde_noise_scaler(x):
        return x * ((x**0.3).exp() + 10.0)

    def ode_noise_scaler(x):
        return x

    noise_scaler = default_er_sde_noise_scaler if noise_scaler is None else noise_scaler
    num_integration_points = 200.0
    point_indice = torch.arange(0, num_integration_points, dtype=torch.float32, device=x.device)

    sigmas = offset_first_sigma_for_snr(sigmas, model_sampling)
    half_log_snrs = sigma_to_half_log_snr(sigmas, model_sampling)
    er_lambdas = half_log_snrs.neg().exp()  # er_lambda_t = sigma_t / alpha_t

    old_denoised = None
    old_denoised_d = None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        stage_used = min(max_stage, i + 1)
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            # sa_solver fragment
            tau_t = tau_func(sigmas[i + 1])
            inject_noise = tau_t > 0 and s_noise > 0

            noise_scaler_i = noise_scaler if inject_noise else ode_noise_scaler

            er_lambda_s, er_lambda_t = er_lambdas[i], er_lambdas[i + 1]
            alpha_s = sigmas[i] / er_lambda_s
            alpha_t = sigmas[i + 1] / er_lambda_t
            r_alpha = alpha_t / alpha_s
            r = noise_scaler_i(er_lambda_t) / noise_scaler_i(er_lambda_s)

            # Stage 1 Euler
            x = r_alpha * r * x + alpha_t * (1 - r) * denoised

            if stage_used >= 2:
                dt = er_lambda_t - er_lambda_s
                lambda_step_size = -dt / num_integration_points
                lambda_pos = er_lambda_t + point_indice * lambda_step_size
                scaled_pos = noise_scaler_i(lambda_pos)

                # Stage 2
                s = torch.sum(1 / scaled_pos) * lambda_step_size
                denoised_d = (denoised - old_denoised) / (er_lambda_s - er_lambdas[i - 1])
                x = x + alpha_t * (dt + s * noise_scaler_i(er_lambda_t)) * denoised_d

                if stage_used >= 3:
                    # Stage 3
                    s_u = torch.sum((lambda_pos - er_lambda_s) / scaled_pos) * lambda_step_size
                    denoised_u = (denoised_d - old_denoised_d) / ((er_lambda_s - er_lambdas[i - 2]) / 2)
                    x = x + alpha_t * ((dt**2) / 2 + s_u * noise_scaler_i(er_lambda_t)) * denoised_u
                old_denoised_d = denoised_d

            if s_noise > 0:
                x = x + alpha_t * noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * (
                    er_lambda_t**2 - er_lambda_s**2 * r**2
                ).sqrt().nan_to_num(nan=0.0)
        old_denoised = denoised
    return x
