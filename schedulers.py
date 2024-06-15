import torch

import comfy.samplers
from comfy.samplers import SCHEDULER_NAMES, simple_scheduler
from comfy_extras.nodes_align_your_steps import loglinear_interp


calculate_sigmas_original = comfy.samplers.calculate_sigmas


# Modified AYS by Extraltodeus from https://github.com/Extraltodeus/sigmas_tools_and_the_golden_scheduler/blob/0dc89a264ef346a093d053c0da751f3ece317613/sigmas_merge.py#L203-L233
def _ays_scheduler(model_sampling, steps, force_sigma_min, model_type="SDXL"):
    timestep_indices = {
        "SD1": [999, 850, 736, 645, 545, 455, 343, 233, 124, 24, 0],
        "SDXL": [999, 845, 730, 587, 443, 310, 193, 116, 53, 13, 0],
        "SVD": [995, 920, 811, 686, 555, 418, 315, 174, 109, 12, 0],
    }
    indices = timestep_indices[model_type]
    indices = [999 - i for i in indices]
    sigmas = simple_scheduler(model_sampling, 1000)[indices]
    sigmas = loglinear_interp(sigmas.tolist(), steps + 1 if not force_sigma_min else steps)
    sigmas = torch.FloatTensor(sigmas)
    sigmas = torch.cat([sigmas[:-1] if not force_sigma_min else sigmas, torch.FloatTensor([0.0])])
    return sigmas.cpu()


def _calculate_sigmas_hijack(model_sampling, scheduler_name, steps):
    if scheduler_name == "ays":
        sigmas = _ays_scheduler(model_sampling, steps, False)
    elif scheduler_name == "ays+":
        sigmas = _ays_scheduler(model_sampling, steps, True)
    else:
        sigmas = calculate_sigmas_original(model_sampling, scheduler_name, steps)
    return sigmas


def hijack_schedulers():
    SCHEDULER_NAMES.append("ays")
    SCHEDULER_NAMES.append("ays+")
    assert calculate_sigmas_original != _calculate_sigmas_hijack
    comfy.samplers.calculate_sigmas = _calculate_sigmas_hijack
