from .latent_ratio import EmptyLatentImageAR
from .token_counter import TokenCounter
from .random_gen import RandomPromptGenerator
from .cascade_utils import StableCascade_AutoCompLatent
from .clip_misc import CLIPTextEncodeBREAK, CLIPMicroConditioning
from .clip_negpip import CLIPNegPip


NODE_CLASS_MAPPINGS = {
    # "CLIPSetLastLayerFloat": CLIPSetLastLayerFloat,
    "EmptyLatentImageAR": EmptyLatentImageAR,
    "TokenCounter": TokenCounter,
    "RandomPromptGenerator": RandomPromptGenerator,
    "StableCascade_AutoCompLatent": StableCascade_AutoCompLatent,
    "CLIPTextEncodeBREAK": CLIPTextEncodeBREAK,
    "CLIPMicroConditioning": CLIPMicroConditioning,
    "CLIPNegPip": CLIPNegPip,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # "CLIPSetLastLayerFloat": "CLIP Set Last Layer (Float)",
    "EmptyLatentImageAR": "Empty Latent Image (Aspect Ratio)",
    "TokenCounter": "Token Counter",
    "RandomPromptGenerator": "Random Prompt Generator",
    "StableCascade_AutoCompLatent": "StableCascade_AutoCompLatent",
    "CLIPTextEncode": "CLIPTextEncode",
    "CLIPMicroConditioning": "CLIPMicroConditioning",
    "CLIPNegPip": "CLIPNegPip",
}
