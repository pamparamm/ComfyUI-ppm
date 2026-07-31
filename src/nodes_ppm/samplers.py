from comfy.comfy_types.node_typing import IO, ComfyNodeABC, InputTypeDict
from comfy.k_diffusion import sampling as k_diffusion_sampling
from comfy.k_diffusion.sa_solver import get_tau_interval_func
from comfy.model_patcher import ModelPatcher
from comfy.samplers import KSAMPLER
from comfy_api.latest import io

from ..sampling import ppm_cfgpp_dyn_sampling, ppm_cfgpp_sampling, ppm_dyn_sampling, ppm_sampling

CFGPP_SAMPLER_NAMES_COMFY_ETA: list = [
    "euler_ancestral_cfg_pp",
]
CFGPP_SAMPLER_NAMES_COMFY: list = [
    "euler_cfg_pp",
    "dpmpp_2m_cfg_pp",
    "gradient_estimation_cfg_pp",
    *CFGPP_SAMPLER_NAMES_COMFY_ETA,
]


CFGPP_SAMPLER_NAMES: list = [
    *CFGPP_SAMPLER_NAMES_COMFY,
    *ppm_cfgpp_sampling.CFGPP_SAMPLER_NAMES_KD,
    *ppm_cfgpp_dyn_sampling.CFGPP_SAMPLER_NAMES_DYN,
]
SAMPLER_NAMES_ETA: list = [
    *CFGPP_SAMPLER_NAMES_COMFY_ETA,
    *ppm_cfgpp_sampling.CFGPP_SAMPLER_NAMES_KD_ETA,
    *ppm_cfgpp_dyn_sampling.CFGPP_SAMPLER_NAMES_DYN_ETA,
    *ppm_dyn_sampling.SAMPLER_NAMES_DYN_ETA,
]


class DynSamplerSelect(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "sampler_name": (IO.COMBO, {"options": ppm_dyn_sampling.SAMPLER_NAMES_DYN}),
                "eta": (IO.FLOAT, {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01, "round": False}),
                "s_dy_pow": (IO.INT, {"default": 2, "min": -1, "max": 100}),
                "s_extra_steps": (IO.BOOLEAN, {"default": False}),
            }
        }

    RETURN_TYPES = (IO.SAMPLER,)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, sampler_name: str, eta=1.0, s_dy_pow=-1, s_extra_steps=False):
        sampler_func = getattr(ppm_dyn_sampling, "sample_{}".format(sampler_name))
        extra_options = {}
        if sampler_name in SAMPLER_NAMES_ETA:
            extra_options["eta"] = eta
        extra_options["s_dy_pow"] = s_dy_pow
        extra_options["s_extra_steps"] = s_extra_steps
        sampler = KSAMPLER(sampler_func, extra_options=extra_options)
        return (sampler,)


# More CFG++ samplers based on https://github.com/comfyanonymous/ComfyUI/pull/3871 by yoinked-h
class CFGPPSamplerSelect(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "sampler_name": (IO.COMBO, {"options": CFGPP_SAMPLER_NAMES}),
                "eta": (IO.FLOAT, {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01, "round": False}),
                "s_gamma_start": (IO.FLOAT, {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.01, "round": False}),
                "s_gamma_end": (IO.FLOAT, {"default": 1.0, "min": 0.0, "max": 10000.0, "step": 0.01, "round": False}),
                "s_extra_steps": (IO.BOOLEAN, {"default": False}),
            }
        }

    RETURN_TYPES = (IO.SAMPLER,)
    CATEGORY = "model/sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, sampler_name: str, eta=1.0, s_gamma_start=0.0, s_gamma_end=1.0, s_extra_steps=False):
        sampler_func = self._get_sampler_func(sampler_name)
        extra_options = {}
        if sampler_name in SAMPLER_NAMES_ETA:
            extra_options["eta"] = eta
        if sampler_name in ppm_cfgpp_dyn_sampling.CFGPP_SAMPLER_NAMES_DYN:
            extra_options["s_gamma_start"] = s_gamma_start
            extra_options["s_gamma_end"] = s_gamma_end
            extra_options["s_extra_steps"] = s_extra_steps
        sampler = KSAMPLER(sampler_func, extra_options=extra_options)
        return (sampler,)

    def _get_sampler_func(self, sampler_name: str):
        if sampler_name in CFGPP_SAMPLER_NAMES_COMFY:
            return getattr(k_diffusion_sampling, "sample_{}".format(sampler_name))
        if sampler_name in ppm_cfgpp_sampling.CFGPP_SAMPLER_NAMES_KD:
            return getattr(ppm_cfgpp_sampling, "sample_{}".format(sampler_name))
        if sampler_name in ppm_cfgpp_dyn_sampling.CFGPP_SAMPLER_NAMES_DYN:
            return getattr(ppm_cfgpp_dyn_sampling, "sample_{}".format(sampler_name))

        raise ValueError(f"Unknown sampler_name {sampler_name}")


class PPMSamplerSelect(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "sampler_name": (IO.COMBO, {"options": ppm_sampling.SAMPLER_NAMES}),
                "model": (IO.MODEL, {}),
                "cfg_pp": (IO.BOOLEAN, {"default": False}),
                "s_sigma_diff": (IO.FLOAT, {"default": 2.0, "min": 0.0, "max": 10000.0, "step": 0.01, "round": False}),
            }
        }

    RETURN_TYPES = (IO.SAMPLER,)
    CATEGORY = "model/sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, sampler_name: str, model: ModelPatcher, cfg_pp=False, s_sigma_diff=2.0):
        sampler_func = getattr(ppm_sampling, "sample_{}".format(sampler_name))
        ms = model.get_model_object("model_sampling")
        extra_options = {}
        extra_options["cfg_pp"] = cfg_pp
        extra_options["s_sigma_diff"] = s_sigma_diff
        extra_options["s_sigma_max"] = ms.sigma_max
        sampler = KSAMPLER(sampler_func, extra_options=extra_options)
        return (sampler,)


class SamplerGradientEstimation(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "sampler_name": (IO.COMBO, {"options": ["gradient_estimation", "gradient_estimation_cfg_pp"]}),
                "gamma": (IO.FLOAT, {"default": 2.0, "min": 2.0, "max": 5.0, "step": 0.01, "round": 0.001}),
            }
        }

    RETURN_TYPES = (IO.SAMPLER,)
    CATEGORY = "model/sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, sampler_name: str, gamma=2.0):
        sampler_func = getattr(k_diffusion_sampling, "sample_{}".format(sampler_name))
        extra_options = {}
        extra_options["ge_gamma"] = gamma
        sampler = KSAMPLER(sampler_func, extra_options=extra_options)
        return (sampler,)


class SamplerSEEDS2Scheduled(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SamplerSEEDS2Scheduled",
            search_aliases=["sde", "exp heun"],
            category="model/sampling/samplers",
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("solver_type", options=["phi_1", "phi_2"]),
                io.Float.Input(
                    "eta",
                    default=1.0,
                    min=0.0,
                    max=100.0,
                    step=0.01,
                    round=False,
                    tooltip="Stochastic strength",
                    advanced=True,
                ),
                io.Float.Input("sde_start_percent", default=0.2, min=0.0, max=1.0, step=0.001, advanced=True),
                io.Float.Input("sde_end_percent", default=1.0, min=0.0, max=1.0, step=0.001, advanced=True),
                io.Float.Input(
                    "s_noise",
                    default=1.0,
                    min=0.0,
                    max=100.0,
                    step=0.01,
                    round=False,
                    tooltip="SDE noise multiplier",
                    advanced=True,
                ),
                io.Float.Input(
                    "r",
                    default=0.5,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    round=False,
                    tooltip="Relative step size for the intermediate stage (c2 node)",
                    advanced=True,
                ),
            ],
            outputs=[io.Sampler.Output()],
            description=(
                "Modified SEEDS 2 sampler with SDE scheduling akin to sa_solver\n"
                "This sampler node can represent multiple samplers:\n\n"
                "seeds_2\n"
                "- default setting\n\n"
                "exp_heun_2_x0\n"
                "- solver_type=phi_2, r=1.0, eta=0.0\n\n"
                "exp_heun_2_x0_sde\n"
                "- solver_type=phi_2, r=1.0, eta=1.0, s_noise=1.0"
            ),
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        model = kwargs["model"]
        solver_type = kwargs["solver_type"]
        eta = kwargs["eta"]
        sde_start_percent = kwargs["sde_start_percent"]
        sde_end_percent = kwargs["sde_end_percent"]
        s_noise = kwargs["s_noise"]
        r = kwargs["r"]

        model_sampling = model.get_model_object("model_sampling")
        start_sigma = model_sampling.percent_to_sigma(sde_start_percent)
        end_sigma = model_sampling.percent_to_sigma(sde_end_percent)
        tau_func = get_tau_interval_func(start_sigma, end_sigma, eta=eta)

        extra_options = {}
        extra_options["eta"] = eta
        extra_options["s_noise"] = s_noise
        extra_options["r"] = r
        extra_options["solver_type"] = solver_type
        extra_options["tau_func"] = tau_func

        sampler_func = ppm_sampling.sample_seeds_2_scheduled

        sampler = KSAMPLER(sampler_func, extra_options=extra_options)
        return io.NodeOutput(sampler)


class SamplerER_SDEScheduled(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SamplerER_SDEScheduled",
            category="model/sampling/samplers",
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("solver_type", options=["ER-SDE", "Reverse-time SDE", "ODE"]),
                io.Int.Input("max_stage", default=3, min=1, max=3, advanced=True),
                io.Float.Input(
                    "eta",
                    default=1.0,
                    min=0.0,
                    max=100.0,
                    step=0.01,
                    round=False,
                    tooltip="Stochastic strength of reverse-time SDE.\nWhen eta=0, it reduces to deterministic ODE. This setting doesn't apply to ER-SDE solver type.",
                    advanced=True,
                ),
                io.Float.Input("sde_start_percent", default=0.2, min=0.0, max=1.0, step=0.001, advanced=True),
                io.Float.Input("sde_end_percent", default=1.0, min=0.0, max=1.0, step=0.001, advanced=True),
                io.Float.Input("s_noise", default=1.0, min=0.0, max=100.0, step=0.01, round=False, advanced=True),
            ],
            outputs=[io.Sampler.Output()],
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        model = kwargs["model"]
        solver_type = kwargs["solver_type"]
        max_stage = kwargs["max_stage"]
        eta = kwargs["eta"]
        sde_start_percent = kwargs["sde_start_percent"]
        sde_end_percent = kwargs["sde_end_percent"]
        s_noise = kwargs["s_noise"]

        model_sampling = model.get_model_object("model_sampling")
        start_sigma = model_sampling.percent_to_sigma(sde_start_percent)
        end_sigma = model_sampling.percent_to_sigma(sde_end_percent)
        tau_func = get_tau_interval_func(start_sigma, end_sigma, eta=eta)

        if solver_type == "ODE" or (solver_type == "Reverse-time SDE" and eta == 0):
            eta = 0
            s_noise = 0

        def reverse_time_sde_noise_scaler(x):
            return x ** (eta + 1)

        if solver_type == "ER-SDE":
            # Use the default one in sample_er_sde()
            noise_scaler = None
        else:
            noise_scaler = reverse_time_sde_noise_scaler

        extra_options = {}
        extra_options["s_noise"] = s_noise
        extra_options["noise_scaler"] = noise_scaler
        extra_options["max_stage"] = max_stage
        extra_options["tau_func"] = tau_func

        sampler_func = ppm_sampling.sample_er_sde_scheduled

        sampler = KSAMPLER(sampler_func, extra_options=extra_options)
        return io.NodeOutput(sampler)


NODE_CLASS_MAPPINGS = {
    "CFGPPSamplerSelect": CFGPPSamplerSelect,
    "DynSamplerSelect": DynSamplerSelect,
    "PPMSamplerSelect": PPMSamplerSelect,
    "SamplerGradientEstimation": SamplerGradientEstimation,
    "SamplerSEEDS2Scheduled": SamplerSEEDS2Scheduled,
    "SamplerER_SDEScheduled": SamplerER_SDEScheduled,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CFGPPSamplerSelect": "CFG++SamplerSelect",
    "DynSamplerSelect": "DynSamplerSelect",
    "PPMSamplerSelect": "PPMSamplerSelect",
    "SamplerGradientEstimation": "SamplerGradientEstimation",
    "SamplerSEEDS2Scheduled": "SamplerSEEDS2Scheduled",
    "SamplerER_SDEScheduled": "SamplerER_SDEScheduled",
}
