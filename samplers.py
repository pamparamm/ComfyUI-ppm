from comfy.k_diffusion import sampling as k_diffusion_sampling
from comfy.samplers import KSAMPLER
from . import ppm_cfgpp_sampling

INITIALIZED = False
CFGPP_SAMPLER_NAMES_ORIGINAL = ["euler_cfg_pp", "euler_ancestral_cfg_pp"]
CFGPP_SAMPLER_NAMES = CFGPP_SAMPLER_NAMES_ORIGINAL + ["dpmpp_2m_cfg_pp", "dpmpp_2m_sde_cfg_pp", "dpmpp_2m_sde_gpu_cfg_pp"]


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
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, sampler_name):
        if sampler_name in CFGPP_SAMPLER_NAMES_ORIGINAL:
            sampler_func = getattr(k_diffusion_sampling, "sample_{}".format(sampler_name))
        else:
            sampler_func = getattr(ppm_cfgpp_sampling, "sample_{}".format(sampler_name))
        sampler = KSAMPLER(sampler_func)
        return (sampler,)
