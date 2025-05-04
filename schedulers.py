import comfy.samplers
import torch
from comfy.samplers import SCHEDULER_NAMES, beta_scheduler, simple_scheduler
from comfy_extras.nodes_align_your_steps import loglinear_interp
from comfy_extras.nodes_gits import GITSScheduler

calculate_sigmas_original = comfy.samplers.calculate_sigmas

AYS_SCHEDULER = "ays"
AYS_PLUS_SCHEDULER = "ays+"
AYS_30_SCHEDULER = "ays_30"
AYS_30_PLUS_SCHEDULER = "ays_30+"
GITS_SCHEDULER = "gits"
BETA_1_1_SCHEDULER = "beta_1_1"


# Modified AYS by Extraltodeus from https://github.com/Extraltodeus/sigmas_tools_and_the_golden_scheduler/blob/0dc89a264ef346a093d053c0da751f3ece317613/sigmas_merge.py#L203-L233
# 30-step AYS by Koitenshin from https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/15751#issuecomment-2143648234
def _ays_scheduler(model_sampling, steps, force_sigma_min, model_type="SDXL"):
    timestep_indices = {
        "SD1": [999, 850, 736, 645, 545, 455, 343, 233, 124, 24, 0],
        "SDXL": [999, 845, 730, 587, 443, 310, 193, 116, 53, 13, 0],
        "SD1_30": [999, 954, 907, 855, 819, 782, 744, 713, 684, 654, 623, 592, 559, 528, 499, 470, 437, 401, 364, 328, 291, 256, 221, 183, 149, 114, 70, 41, 21, 4, 0],
        "SDXL_30": [999, 953, 904, 850, 813, 777, 738, 695, 650, 602, 556, 510, 462, 417, 374, 331, 290, 250, 214, 182, 155, 131, 108, 85, 66, 49, 32, 20, 12, 3, 0],
    }  # fmt: skip
    indices = timestep_indices[model_type]
    sigmas = simple_scheduler(model_sampling, 1000).flip(0)[1:][indices]
    sigmas = loglinear_interp(sigmas.tolist(), steps + 1 if not force_sigma_min else steps)
    sigmas = torch.FloatTensor(sigmas)
    sigmas = torch.cat([sigmas[:-1] if not force_sigma_min else sigmas, torch.FloatTensor([0.0])])
    return sigmas.cpu()


def _calculate_sigmas_hijack(model_sampling, scheduler_name, steps):
    if scheduler_name == AYS_SCHEDULER:
        sigmas = _ays_scheduler(model_sampling, steps, False)
    elif scheduler_name == AYS_PLUS_SCHEDULER:
        sigmas = _ays_scheduler(model_sampling, steps, True)
    elif scheduler_name == AYS_30_SCHEDULER:
        sigmas = _ays_scheduler(model_sampling, steps, False, "SDXL_30")
    elif scheduler_name == AYS_30_PLUS_SCHEDULER:
        sigmas = _ays_scheduler(model_sampling, steps, True, "SDXL_30")
    elif scheduler_name == GITS_SCHEDULER:
        sigmas = GITSScheduler().get_sigmas(1.2, steps, 1.0)[0]
    elif scheduler_name == BETA_1_1_SCHEDULER:
        sigmas = beta_scheduler(model_sampling, steps, 1.0, 1.0)
    else:
        sigmas = calculate_sigmas_original(model_sampling, scheduler_name, steps)
    return sigmas


def hijack_schedulers():
    if calculate_sigmas_original == _calculate_sigmas_hijack:
        raise RuntimeError("Schedulers are already hijacked")

    SCHEDULER_NAMES.append(AYS_SCHEDULER)
    SCHEDULER_NAMES.append(AYS_PLUS_SCHEDULER)
    SCHEDULER_NAMES.append(AYS_30_SCHEDULER)
    SCHEDULER_NAMES.append(AYS_30_PLUS_SCHEDULER)
    SCHEDULER_NAMES.append(GITS_SCHEDULER)
    SCHEDULER_NAMES.append(BETA_1_1_SCHEDULER)

    comfy.samplers.calculate_sigmas = _calculate_sigmas_hijack
