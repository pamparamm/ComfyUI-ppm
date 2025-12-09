from functools import partial
import torch

import comfy.samplers
from comfy.samplers import SchedulerHandler, beta_scheduler, simple_scheduler
from comfy_extras.nodes_align_your_steps import loglinear_interp
from comfy_extras.nodes_gits import NOISE_LEVELS as GITS_NOISE_LEVELS


# Modified AYS by Extraltodeus from https://github.com/Extraltodeus/sigmas_tools_and_the_golden_scheduler/blob/0dc89a264ef346a093d053c0da751f3ece317613/sigmas_merge.py#L203-L233
# 30-step AYS by Koitenshin from https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/15751#issuecomment-2143648234
def _ays_scheduler(model_sampling, steps, force_sigma_min=False, model_type="SDXL"):
    timestep_indices = {
        "SDXL": [999, 845, 730, 587, 443, 310, 193, 116, 53, 13, 0],
        "SDXL_30": [999, 953, 904, 850, 813, 777, 738, 695, 650, 602, 556, 510, 462, 417, 374, 331, 290, 250, 214, 182, 155, 131, 108, 85, 66, 49, 32, 20, 12, 3, 0],
    }  # fmt: skip
    indices = timestep_indices[model_type]
    sigmas = simple_scheduler(model_sampling, 1000).flip(0)[1:][indices]
    sigmas = loglinear_interp(sigmas.tolist(), steps + 1 if not force_sigma_min else steps)
    sigmas = torch.FloatTensor(sigmas)
    sigmas = torch.cat([sigmas[:-1] if not force_sigma_min else sigmas, torch.FloatTensor([0.0])])
    return torch.FloatTensor(sigmas)


# Copied from comfy_extras.nodes_gits.GITSScheduler
def _gits_scheduler(model_sampling, steps, coeff=1.2):
    if steps <= 20:
        sigmas = GITS_NOISE_LEVELS[round(coeff, 2)][steps - 2][:]
    else:
        sigmas = GITS_NOISE_LEVELS[round(coeff, 2)][-1][:]
        sigmas = loglinear_interp(sigmas, steps + 1)

    sigmas = sigmas[-(steps + 1) :]
    sigmas[-1] = 0
    return torch.FloatTensor(sigmas)


CUSTOM_HANDLERS = {
    "ays": SchedulerHandler(partial(_ays_scheduler, force_sigma_min=False, model_type="SDXL")),
    "ays+": SchedulerHandler(partial(_ays_scheduler, force_sigma_min=True, model_type="SDXL_30")),
    "ays_30": SchedulerHandler(partial(_ays_scheduler, force_sigma_min=False, model_type="SDXL")),
    "ays_30+": SchedulerHandler(partial(_ays_scheduler, force_sigma_min=True, model_type="SDXL_30")),
    "gits": SchedulerHandler(partial(_gits_scheduler, coeff=1.2)),
    "beta_1_1": SchedulerHandler(partial(beta_scheduler, alpha=1.0, beta=1.0)),
}


def inject_schedulers():
    if any(key in comfy.samplers.SCHEDULER_HANDLERS for key in CUSTOM_HANDLERS):
        raise RuntimeError("Schedulers are already injected")

    comfy.samplers.SCHEDULER_HANDLERS.update(CUSTOM_HANDLERS)
    comfy.samplers.SCHEDULER_NAMES.extend(CUSTOM_HANDLERS)
