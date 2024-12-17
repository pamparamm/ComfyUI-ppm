from .nodes.latent_misc import EmptyLatentImageAR, LatentToWidthHeight, LatentToMaskBB
from .nodes.latent_tonemap import LatentOperationTonemapLuminance
from .nodes.clip_misc import CLIPTextEncodeBREAK, CLIPMicroConditioning, CLIPTokenCounter, ConditioningZeroOutCombine
from .nodes.clip_negpip import CLIPNegPip
from .nodes.attention_couple_ppm import AttentionCouplePPM
from .nodes.guidance_limiter import GuidanceLimiter, CFGLimiterGuider
from .nodes.samplers import CFGPPSamplerSelect, DynSamplerSelect, PPMSamplerSelect
from .nodes.freeu_adv import FreeU2PPM
from .nodes.misc import ConvertTimestepToSigma

from .schedulers import hijack_schedulers
from .compat.advanced_encode import hijack_adv_encode


WEB_DIRECTORY = "./js"

NODE_CLASS_MAPPINGS = {
    "EmptyLatentImageAR": EmptyLatentImageAR,
    "LatentToWidthHeight": LatentToWidthHeight,
    "LatentToMaskBB": LatentToMaskBB,
    "LatentOperationTonemapLuminance": LatentOperationTonemapLuminance,
    "CLIPTextEncodeBREAK": CLIPTextEncodeBREAK,
    "CLIPMicroConditioning": CLIPMicroConditioning,
    "CLIPTokenCounter": CLIPTokenCounter,
    "ConditioningZeroOutCombine": ConditioningZeroOutCombine,
    "CLIPNegPip": CLIPNegPip,
    "AttentionCouplePPM": AttentionCouplePPM,
    "Guidance Limiter": GuidanceLimiter,
    "CFGLimiterGuider": CFGLimiterGuider,
    "CFGPPSamplerSelect": CFGPPSamplerSelect,
    "DynSamplerSelect": DynSamplerSelect,
    "PPMSamplerSelect": PPMSamplerSelect,
    "FreeU2PPM": FreeU2PPM,
    "ConvertTimestepToSigma": ConvertTimestepToSigma,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EmptyLatentImageAR": "Empty Latent Image (Aspect Ratio)",
    "LatentToWidthHeight": "Latent to Width & Height",
    "LatentToMaskBB": "Latent to Mask (Bounding Box)",
    "LatentOperationTonemapLuminance": "LatentOperationTonemapLuminance",
    "CLIPTextEncodeBREAK": "CLIP Text Encode (BREAK)",
    "CLIPMicroConditioning": "CLIPMicroConditioning",
    "CLIPTokenCounter": "CLIPTokenCounter",
    "ConditioningZeroOutCombine": "ConditioningZeroOut (Combine)",
    "CLIPNegPip": "CLIP NegPip",
    "AttentionCouplePPM": "Attention Couple (PPM)",
    "Guidance Limiter": "Guidance Limiter",
    "CFGLimiterGuider": "CFGLimiterGuider",
    "CFGPPSamplerSelect": "CFG++SamplerSelect",
    "DynSamplerSelect": "DynSamplerSelect",
    "PPMSamplerSelect": "PPMSamplerSelect",
    "FreeU2PPM": "FreeU V2 (PPM)",
    "ConvertTimestepToSigma": "Convert Timestep To Sigma",
}


hijack_schedulers()

hijack_adv_encode()
