# Modified samplers from Euler-Smea-Dyn-Sampler by Koishi-Star
from tqdm.auto import trange
import torch

from comfy.k_diffusion.sampling import to_d, default_noise_sampler, get_ancestral_step, BrownianTreeNoiseSampler

SAMPLER_NAMES_DYN_ETA = [
    "euler_ancestral_dy",
]
SAMPLER_NAMES_DYN = [
    "euler_dy",
    "euler_smea_dy",
    "dpmpp_2m_dy",
    "dpmpp_3m_dy",
    *SAMPLER_NAMES_DYN_ETA,
]


class Rescaler:
    def __init__(self, model, x, mode, **extra_args):
        self.model = model
        self.x = x
        self.mode = mode
        self.extra_args = extra_args

        self.latent_image, self.noise = model.latent_image, model.noise
        self.denoise_mask = self.extra_args.get("denoise_mask", None)

    def __enter__(self):
        if self.latent_image is not None:
            self.model.latent_image = torch.nn.functional.interpolate(input=self.latent_image, size=self.x.shape[2:4], mode=self.mode)
        if self.noise is not None:
            self.model.noise = torch.nn.functional.interpolate(input=self.latent_image, size=self.x.shape[2:4], mode=self.mode)
        if self.denoise_mask is not None:
            self.extra_args["denoise_mask"] = torch.nn.functional.interpolate(input=self.denoise_mask, size=self.x.shape[2:4], mode=self.mode)

        return self

    def __exit__(self, type, value, traceback):
        del self.model.latent_image, self.model.noise
        self.model.latent_image, self.model.noise = self.latent_image, self.noise


@torch.no_grad()
def dy_sampling_step(x, model, dt, i, sigma, sigma_hat, callback, **extra_args):
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

    d = to_d(c, sigma_hat, denoised)
    c = c + d * dt

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
def sample_euler_dy(
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
        d = to_d(x, sigma_hat, denoised)
        # Euler method
        x = x + d * dt
        if sigmas[i + 1] > 0 and s_extra_steps:
            if i // 2 == 1:
                x = dy_sampling_step(x, model, dt, i, sigmas[i], sigma_hat, callback, **extra_args)
    return x


@torch.no_grad()
def smea_sampling_step(x, model, dt, i, sigma, sigma_hat, callback, **extra_args):
    m, n = x.shape[2], x.shape[3]
    x = torch.nn.functional.interpolate(input=x, scale_factor=(1.25, 1.25), mode="nearest-exact")

    with Rescaler(model, x, "nearest-exact", **extra_args) as rescaler:
        denoised = model(x, sigma_hat * x.new_ones([x.shape[0]]), **rescaler.extra_args)
    if callback is not None:
        callback({"x": x, "i": i, "sigma": sigma, "sigma_hat": sigma_hat, "denoised": denoised})

    d = to_d(x, sigma_hat, denoised)
    x = x + d * dt
    x = torch.nn.functional.interpolate(input=x, size=(m, n), mode="nearest-exact")
    return x


@torch.no_grad()
def sample_euler_smea_dy(
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
        d = to_d(x, sigma_hat, denoised)
        # Euler method
        x = x + d * dt
        if sigmas[i + 1] > 0 and s_extra_steps:
            if i + 1 // 2 == 1:
                x = dy_sampling_step(x, model, dt, i, sigmas[i], sigma_hat, callback, **extra_args)
            if i + 1 // 2 == 0:
                x = smea_sampling_step(x, model, dt, i, sigmas[i], sigma_hat, callback, **extra_args)
    return x


@torch.no_grad()
def sample_euler_ancestral_dy(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
    s_dy_pow=-1,
    s_extra_steps=True,
):
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = 2**0.5 - 1
        if s_dy_pow >= 0:
            gamma = gamma * (1.0 - (i / (len(sigmas) - 2)) ** s_dy_pow)
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x - eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5

        denoised = model(x, sigma_hat * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigma_hat, sigmas[i + 1], eta=eta)

        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})
        d = to_d(x, sigma_hat, denoised)
        # Euler method
        dt = sigma_down - sigma_hat
        x = x + d * dt
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigma_hat, sigmas[i + 1]) * s_noise * sigma_up
    return x


@torch.no_grad()
def sample_dpmpp_2m_sde_dy(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
    solver_type="midpoint",
    s_dy_pow=-1,
    s_extra_steps=True,
):
    """DPM-Solver++(2M) SDE."""
    if len(sigmas) <= 1:
        return x

    if solver_type not in {"heun", "midpoint"}:
        raise ValueError("solver_type must be 'heun' or 'midpoint'")

    seed = extra_args.get("seed", None)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    old_denoised = None
    h_last = None
    h = None

    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = 2**0.5 - 1
        if s_dy_pow >= 0:
            gamma = gamma * (1.0 - (i / (len(sigmas) - 2)) ** s_dy_pow)
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x - eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            # DPM-Solver++(2M) SDE
            t, s = -sigma_hat.log(), -sigmas[i + 1].log()
            h = s - t
            eta_h = eta * h

            x = sigmas[i + 1] / sigma_hat * (-eta_h).exp() * x + (-h - eta_h).expm1().neg() * denoised

            if old_denoised is not None:
                r = h_last / h
                if solver_type == "heun":
                    x = x + ((-h - eta_h).expm1().neg() / (-h - eta_h) + 1) * (1 / r) * (denoised - old_denoised)
                elif solver_type == "midpoint":
                    x = x + 0.5 * (-h - eta_h).expm1().neg() * (1 / r) * (denoised - old_denoised)

            # TODO not working properly
            if eta:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * eta_h).expm1().neg().sqrt() * s_noise

        old_denoised = denoised
        h_last = h
    return x


@torch.no_grad()
def sample_dpmpp_3m_sde_dy(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
    s_dy_pow=-1,
    s_extra_steps=True,
):
    """DPM-Solver++(3M) SDE."""

    if len(sigmas) <= 1:
        return x

    seed = extra_args.get("seed", None)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    denoised_1, denoised_2 = None, None
    h, h_1, h_2 = None, None, None

    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = 2**0.5 - 1
        if s_dy_pow >= 0:
            gamma = gamma * (1.0 - (i / (len(sigmas) - 2)) ** s_dy_pow)
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x - eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            t, s = -sigma_hat.log(), -sigmas[i + 1].log()
            h = s - t
            h_eta = h * (eta + 1)

            x = torch.exp(-h_eta) * x + (-h_eta).expm1().neg() * denoised

            if h_2 is not None:
                r0 = h_1 / h
                r1 = h_2 / h
                d1_0 = (denoised - denoised_1) / r0
                d1_1 = (denoised_1 - denoised_2) / r1
                d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                d2 = (d1_0 - d1_1) / (r0 + r1)
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                phi_3 = phi_2 / h_eta - 0.5
                x = x + phi_2 * d1 - phi_3 * d2
            elif h_1 is not None:
                r = h_1 / h
                d = (denoised - denoised_1) / r
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                x = x + phi_2 * d

            # TODO not working properly
            if eta:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * h * eta).expm1().neg().sqrt() * s_noise

        denoised_1, denoised_2 = denoised, denoised_1
        h_1, h_2 = h, h_1
    return x


@torch.no_grad()
def sample_dpmpp_2m_dy(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    s_noise=1.0,
    noise_sampler=None,
    solver_type="midpoint",
    s_dy_pow=-1,
    s_extra_steps=True,
):
    return sample_dpmpp_2m_sde_dy(
        model,
        x,
        sigmas,
        extra_args,
        callback,
        disable,
        0.0,
        s_noise,
        noise_sampler,
        solver_type,
        s_dy_pow,
        s_extra_steps,
    )


@torch.no_grad()
def sample_dpmpp_3m_dy(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    s_noise=1.0,
    noise_sampler=None,
    s_dy_pow=-1,
    s_extra_steps=True,
):
    return sample_dpmpp_3m_sde_dy(
        model,
        x,
        sigmas,
        extra_args,
        callback,
        disable,
        0.0,
        s_noise,
        noise_sampler,
        s_dy_pow,
        s_extra_steps,
    )
