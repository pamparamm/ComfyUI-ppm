from .nodes_ppm.attention_couple_ppm import AttentionCouplePPM
from .nodes_ppm.clip_misc import (
    CLIPMicroConditioning,
    CLIPTextEncodeBREAK,
    CLIPTokenCounter,
    ConditioningZeroOutCombine,
)
from .nodes_ppm.clip_negpip import CLIPNegPip
from .nodes_ppm.freeu_adv import FreeU2PPM
from .nodes_ppm.guidance import (
    CFGLimiterGuider,
    DynamicThresholdingPost,
    DynamicThresholdingSimplePost,
    GuidanceLimiter,
    RenormCFGPost,
    RescaleCFGPost,
)
from .nodes_ppm.latent_misc import (
    EmptyLatentImageAR,
    LatentToMaskBB,
    LatentToWidthHeight,
    MaskCompositePPM,
)
from .nodes_ppm.latent_tonemap import LatentOperationTonemapLuminance
from .nodes_ppm.misc import ConvertTimestepToSigma
from .nodes_ppm.samplers import CFGPPSamplerSelect, DynSamplerSelect, PPMSamplerSelect
from .schedulers import hijack_schedulers

WEB_DIRECTORY = "./js"

NODE_CLASS_MAPPINGS = {
    "EmptyLatentImageAR": EmptyLatentImageAR,
    "LatentToWidthHeight": LatentToWidthHeight,
    "LatentToMaskBB": LatentToMaskBB,
    "MaskCompositePPM": MaskCompositePPM,
    "LatentOperationTonemapLuminance": LatentOperationTonemapLuminance,
    "CLIPTextEncodeBREAK": CLIPTextEncodeBREAK,
    "CLIPMicroConditioning": CLIPMicroConditioning,
    "CLIPTokenCounter": CLIPTokenCounter,
    "ConditioningZeroOutCombine": ConditioningZeroOutCombine,
    "CLIPNegPip": CLIPNegPip,
    "AttentionCouplePPM": AttentionCouplePPM,
    "Guidance Limiter": GuidanceLimiter,
    "CFGLimiterGuider": CFGLimiterGuider,
    "RescaleCFGPost": RescaleCFGPost,
    "DynamicThresholdingSimplePost": DynamicThresholdingSimplePost,
    "DynamicThresholdingPost": DynamicThresholdingPost,
    "RenormCFGPost": RenormCFGPost,
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
    "MaskCompositePPM": "MaskCompositePPM",
    "LatentOperationTonemapLuminance": "LatentOperationTonemapLuminance",
    "CLIPTextEncodeBREAK": "CLIP Text Encode (BREAK)",
    "CLIPMicroConditioning": "CLIPMicroConditioning",
    "CLIPTokenCounter": "CLIPTokenCounter",
    "ConditioningZeroOutCombine": "ConditioningZeroOut (Combine)",
    "CLIPNegPip": "CLIP NegPip",
    "AttentionCouplePPM": "Attention Couple (PPM)",
    "Guidance Limiter": "Guidance Limiter",
    "CFGLimiterGuider": "CFGLimiterGuider",
    "RescaleCFGPost": "RescaleCFGPost",
    "DynamicThresholdingSimplePost": "DynamicThresholdingSimplePost",
    "DynamicThresholdingPost": "DynamicThresholdingFullPost",
    "RenormCFGPost": "RenormCFGPost",
    "CFGPPSamplerSelect": "CFG++SamplerSelect",
    "DynSamplerSelect": "DynSamplerSelect",
    "PPMSamplerSelect": "PPMSamplerSelect",
    "FreeU2PPM": "FreeU V2 (PPM)",
    "ConvertTimestepToSigma": "Convert Timestep To Sigma",
}


hijack_schedulers()
