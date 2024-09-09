from comfy.k_diffusion import sampling as k_diffusion_sampling
from comfy.samplers import KSAMPLER
from ..sampling import ppm_dyn_sampling, ppm_cfgpp_sampling, ppm_cfgpp_dyn_sampling

CFGPP_SAMPLER_NAMES_ORIGINAL_ETA = [
    "euler_ancestral_cfg_pp",
]
CFGPP_SAMPLER_NAMES_ORIGINAL = [
    "euler_cfg_pp",
    *CFGPP_SAMPLER_NAMES_ORIGINAL_ETA,
]


CFGPP_SAMPLER_NAMES = [
    *CFGPP_SAMPLER_NAMES_ORIGINAL,
    *ppm_cfgpp_sampling.CFGPP_SAMPLER_NAMES_KD,
    *ppm_cfgpp_dyn_sampling.CFGPP_SAMPLER_NAMES_DYN,
]
SAMPLER_NAMES_ETA = [
    *CFGPP_SAMPLER_NAMES_ORIGINAL_ETA,
    *ppm_cfgpp_sampling.CFGPP_SAMPLER_NAMES_KD_ETA,
    *ppm_cfgpp_dyn_sampling.CFGPP_SAMPLER_NAMES_DYN_ETA,
    *ppm_dyn_sampling.SAMPLER_NAMES_DYN_ETA,
]


class DynSamplerSelect:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sampler_name": (ppm_dyn_sampling.SAMPLER_NAMES_DYN,),
                "s_dy_pow": ("INT", {"default": 2, "min": -1, "max": 100}),
                "s_extra_steps": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, sampler_name, s_dy_pow=-1, s_extra_steps=False):
        sampler_func = getattr(ppm_dyn_sampling, "sample_{}".format(sampler_name))
        extra_options = {}
        extra_options["s_dy_pow"] = s_dy_pow
        extra_options["s_extra_steps"] = s_extra_steps
        sampler = KSAMPLER(sampler_func, extra_options=extra_options)
        return (sampler,)


# More CFG++ samplers based on https://github.com/comfyanonymous/ComfyUI/pull/3871 by yoinked-h
class CFGPPSamplerSelect:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sampler_name": (CFGPP_SAMPLER_NAMES,),
                "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01, "round": False}),
                "s_dy_pow": ("INT", {"default": 2, "min": -1, "max": 100}),
                "s_extra_steps": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, sampler_name: str, eta=1.0, s_dy_pow=-1, s_extra_steps=False):
        sampler_func = self._get_sampler_func(sampler_name)
        extra_options = {}
        if sampler_name in SAMPLER_NAMES_ETA:
            extra_options["eta"] = eta
        if sampler_name in ppm_cfgpp_dyn_sampling.CFGPP_SAMPLER_NAMES_DYN:
            extra_options["s_dy_pow"] = s_dy_pow
            extra_options["s_extra_steps"] = s_extra_steps
        sampler = KSAMPLER(sampler_func, extra_options=extra_options)
        return (sampler,)

    def _get_sampler_func(self, sampler_name: str):
        if sampler_name in CFGPP_SAMPLER_NAMES_ORIGINAL:
            return getattr(k_diffusion_sampling, "sample_{}".format(sampler_name))
        if sampler_name in ppm_cfgpp_sampling.CFGPP_SAMPLER_NAMES_KD:
            return getattr(ppm_cfgpp_sampling, "sample_{}".format(sampler_name))
        if sampler_name in ppm_cfgpp_dyn_sampling.CFGPP_SAMPLER_NAMES_DYN:
            return getattr(ppm_cfgpp_dyn_sampling, "sample_{}".format(sampler_name))

        raise ValueError(f"Unknown sampler_name {sampler_name}")
