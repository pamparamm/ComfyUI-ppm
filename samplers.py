from comfy.k_diffusion import sampling as k_diffusion_sampling
from comfy.samplers import KSAMPLER
from . import ppm_cfgpp_sampling
from . import ppm_cfgpp_dyn_sampling

INITIALIZED = False
CFGPP_SAMPLER_NAMES_ORIGINAL = ["euler_cfg_pp", "euler_ancestral_cfg_pp"]
CFGPP_SAMPLER_NAMES_ORIGINAL_ETA = ["euler_ancestral_cfg_pp"]


CFGPP_SAMPLER_NAMES = [
    *CFGPP_SAMPLER_NAMES_ORIGINAL,
    *ppm_cfgpp_sampling.CFGPP_SAMPLER_NAMES_KD,
    *ppm_cfgpp_dyn_sampling.CFGPP_SAMPLER_NAMES_DYN,
]
CFGPP_SAMPLER_NAMES_ETA = [
    *CFGPP_SAMPLER_NAMES_ORIGINAL_ETA,
    *ppm_cfgpp_sampling.CFGPP_SAMPLER_NAMES_KD_ETA,
    *ppm_cfgpp_dyn_sampling.CFGPP_SAMPLER_NAMES_DYN_ETA,
]


def inject_samplers():
    global INITIALIZED
    if not INITIALIZED:
        INITIALIZED = True


# More CFG++ samplers based on https://github.com/comfyanonymous/ComfyUI/pull/3871 by yoinked-h
class CFGPPSamplerSelect:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sampler_name": (CFGPP_SAMPLER_NAMES,),
                "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01, "round": False}),
                "s_dy_pow": ("INT", {"default": -1, "min": -1, "max": 100}),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, sampler_name, eta: float, s_dy_pow: int):
        if sampler_name in CFGPP_SAMPLER_NAMES_ORIGINAL:
            sampler_func = getattr(k_diffusion_sampling, "sample_{}".format(sampler_name))
        elif sampler_name in ppm_cfgpp_sampling.CFGPP_SAMPLER_NAMES_KD:
            sampler_func = getattr(ppm_cfgpp_sampling, "sample_{}".format(sampler_name))
        elif sampler_name in ppm_cfgpp_dyn_sampling.CFGPP_SAMPLER_NAMES_DYN:
            sampler_func = getattr(ppm_cfgpp_dyn_sampling, "sample_{}".format(sampler_name))
        extra_options = {}
        if sampler_name in CFGPP_SAMPLER_NAMES_ETA:
            extra_options["eta"] = eta
        if sampler_name in ppm_cfgpp_dyn_sampling.CFGPP_SAMPLER_NAMES_DYN:
            extra_options["s_dy_pow"] = s_dy_pow
        sampler = KSAMPLER(sampler_func, extra_options=extra_options)
        return (sampler,)
