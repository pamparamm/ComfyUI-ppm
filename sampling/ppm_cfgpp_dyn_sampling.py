# Modified samplers from Euler-Smea-Dyn-Sampler by Koishi-Star
from tqdm.auto import trange
import torch

from .ppm_dyn_sampling import Rescaler
from comfy.k_diffusion.sampling import to_d
import comfy.model_patcher

CFGPP_SAMPLER_NAMES_DYN_ETA = []
CFGPP_SAMPLER_NAMES_DYN = [
    "euler_dy_cfg_pp",
    "euler_smea_dy_cfg_pp",
    *CFGPP_SAMPLER_NAMES_DYN_ETA,
]


@torch.no_grad()
def dy_sampling_step_cfg_pp(x, model, sigma_next, i, sigma, sigma_hat, callback, **extra_args):
    temp = [0]

    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(
        model_options, post_cfg_function, disable_cfg1_optimization=True
    )

    original_shape = x.shape
    batch_size, channels, m, n = original_shape[0], original_shape[1], original_shape[2] // 2, original_shape[3] // 2
    extra_row = x.shape[2] % 2 == 1
    extra_col = x.shape[3] % 2 == 1

    if extra_row:
        extra_row_content = x[:, :, -1:, :]
        x = x[:, :, :-1, :]
    if extra_col:
        extra_col_content = x[:, :, :, -1:]
        x = x[:, :, :, :-1]

    a_list = x.unfold(2, 2, 2).unfold(3, 2, 2).contiguous().view(batch_size, channels, m * n, 2, 2)
    c = a_list[:, :, :, 1, 1].view(batch_size, channels, m, n)

    with Rescaler(model, c, "nearest-exact", **extra_args) as rescaler:
        denoised = model(c, sigma_hat * c.new_ones([c.shape[0]]), **rescaler.extra_args)
    if callback is not None:
        callback({"x": c, "i": i, "sigma": sigma, "sigma_hat": sigma_hat, "denoised": denoised})

    d = to_d(c, sigma_hat, temp[0])
    c = denoised + d * sigma_next

    d_list = c.view(batch_size, channels, m * n, 1, 1)
    a_list[:, :, :, 1, 1] = d_list[:, :, :, 0, 0]
    x = a_list.view(batch_size, channels, m, n, 2, 2).permute(0, 1, 2, 4, 3, 5).reshape(batch_size, channels, 2 * m, 2 * n)

    if extra_row or extra_col:
        x_expanded = torch.zeros(original_shape, dtype=x.dtype, device=x.device)
        x_expanded[:, :, : 2 * m, : 2 * n] = x
        if extra_row:
            x_expanded[:, :, -1:, : 2 * n + 1] = extra_row_content
        if extra_col:
            x_expanded[:, :, : 2 * m, -1:] = extra_col_content
        if extra_row and extra_col:
            x_expanded[:, :, -1:, -1:] = extra_col_content[:, :, -1:, :]
        x = x_expanded

    return x


@torch.no_grad()
def sample_euler_dy_cfg_pp(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
    s_dy_pow=-1,
    s_extra_steps=True,
):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    temp = [0]

    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(
        model_options, post_cfg_function, disable_cfg1_optimization=True
    )

    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = max(s_churn / (len(sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.0
        if s_dy_pow >= 0:
            gamma = gamma * (1.0 - (i / (len(sigmas) - 2)) ** s_dy_pow)
        sigma_hat = sigmas[i] * (gamma + 1)
        # print(sigma_hat)
        dt = sigmas[i + 1] - sigma_hat
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x - eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})
        d = to_d(x, sigma_hat, temp[0])
        # Euler method
        x = denoised + d * sigmas[i + 1]
        if sigmas[i + 1] > 0 and s_extra_steps:
            if i // 2 == 1:
                x = dy_sampling_step_cfg_pp(x, model, sigmas[i + 1], i, sigmas[i], sigma_hat, callback, **extra_args)
    return x


@torch.no_grad()
def smea_sampling_step_cfg_pp(x, model, sigma_next, i, sigma, sigma_hat, callback, **extra_args):
    temp = [0]

    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(
        model_options, post_cfg_function, disable_cfg1_optimization=True
    )

    m, n = x.shape[2], x.shape[3]
    x = torch.nn.functional.interpolate(input=x, scale_factor=(1.25, 1.25), mode="nearest-exact")

    with Rescaler(model, x, "nearest-exact", **extra_args) as rescaler:
        denoised = model(x, sigma_hat * x.new_ones([x.shape[0]]), **rescaler.extra_args)
    if callback is not None:
        callback({"x": x, "i": i, "sigma": sigma, "sigma_hat": sigma_hat, "denoised": denoised})

    d = to_d(x, sigma_hat, temp[0])
    x = denoised + d * sigma_next
    x = torch.nn.functional.interpolate(input=x, size=(m, n), mode="nearest-exact")
    return x


@torch.no_grad()
def sample_euler_smea_dy_cfg_pp(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
    s_dy_pow=-1,
    s_extra_steps=True,
):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    temp = [0]

    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(
        model_options, post_cfg_function, disable_cfg1_optimization=True
    )

    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = max(s_churn / (len(sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.0
        if s_dy_pow >= 0:
            gamma = gamma * (1.0 - (i / (len(sigmas) - 2)) ** s_dy_pow)
        sigma_hat = sigmas[i] * (gamma + 1)
        dt = sigmas[i + 1] - sigma_hat
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x - eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})
        d = to_d(x, sigma_hat, temp[0])
        # Euler method
        x = denoised + d * sigmas[i + 1]
        if sigmas[i + 1] > 0 and s_extra_steps:
            if i + 1 // 2 == 1:
                x = dy_sampling_step_cfg_pp(x, model, sigmas[i + 1], i, sigmas[i], sigma_hat, callback, **extra_args)
            if i + 1 // 2 == 0:
                x = smea_sampling_step_cfg_pp(x, model, sigmas[i + 1], i, sigmas[i], sigma_hat, callback, **extra_args)
    return x
