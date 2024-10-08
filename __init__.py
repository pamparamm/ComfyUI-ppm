from .nodes.latent_misc import EmptyLatentImageAR, EmptyLatentImageARAdvanced, LatentToWidthHeight, LatentToMaskBB
from .nodes.random_gen import RandomPromptGenerator
from .nodes.cascade_utils import StableCascade_AutoCompLatent
from .nodes.clip_misc import CLIPTextEncodeBREAK, CLIPMicroConditioning, CLIPTokenCounter
from .nodes.clip_negpip import CLIPNegPip
from .nodes.attention_couple_ppm import AttentionCouplePPM
from .nodes.guidance_limiter import GuidanceLimiter, CFGLimiterGuider
from .nodes.samplers import CFGPPSamplerSelect, DynSamplerSelect

from .schedulers import hijack_schedulers
from .compat.advanced_encode import hijack_adv_encode


WEB_DIRECTORY = "./js"

NODE_CLASS_MAPPINGS = {
    "EmptyLatentImageAR": EmptyLatentImageAR,
    "EmptyLatentImageARAdvanced": EmptyLatentImageARAdvanced,
    "LatentToWidthHeight": LatentToWidthHeight,
    "LatentToMaskBB": LatentToMaskBB,
    "RandomPromptGenerator": RandomPromptGenerator,
    "StableCascade_AutoCompLatent": StableCascade_AutoCompLatent,
    "CLIPTextEncodeBREAK": CLIPTextEncodeBREAK,
    "CLIPMicroConditioning": CLIPMicroConditioning,
    "CLIPTokenCounter": CLIPTokenCounter,
    "CLIPNegPip": CLIPNegPip,
    "AttentionCouplePPM": AttentionCouplePPM,
    "Guidance Limiter": GuidanceLimiter,
    "CFGLimiterGuider": CFGLimiterGuider,
    "CFGPPSamplerSelect": CFGPPSamplerSelect,
    "DynSamplerSelect": DynSamplerSelect,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EmptyLatentImageAR": "Empty Latent Image (Aspect Ratio)",
    "EmptyLatentImageARAdvanced": "Empty Latent Image (Aspect Ratio+)",
    "LatentToWidthHeight": "LatentToWidthHeight",
    "LatentToMaskBB": "LatentToMaskBB",
    "TokenCounter": "Token Counter",
    "RandomPromptGenerator": "Random Prompt Generator",
    "StableCascade_AutoCompLatent": "StableCascade_AutoCompLatent",
    "CLIPTextEncodeBREAK": "CLIPTextEncodeBREAK",
    "CLIPMicroConditioning": "CLIPMicroConditioning",
    "CLIPTokenCounter": "CLIPTokenCounter",
    "CLIPNegPip": "CLIPNegPip",
    "AttentionCouplePPM": "AttentionCouplePPM",
    "Guidance Limiter": "Guidance Limiter",
    "CFGLimiterGuider": "CFGLimiterGuider",
    "CFGPPSamplerSelect": "CFG++SamplerSelect",
    "DynSamplerSelect": "DynSamplerSelect",
}


hijack_schedulers()

hijack_adv_encode()
