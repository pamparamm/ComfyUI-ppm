from typing import Callable

from comfy.comfy_types.node_typing import IO, ComfyNodeABC, InputTypeDict
from comfy.ldm.modules import attention as attn
from comfy.model_patcher import ModelPatcher
from comfy.sd import CLIP


class ModelAttentionSelector(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "model": (IO.MODEL, {}),
                "attention": (
                    IO.COMBO,
                    {
                        "default": cls.ATTENTION_OPTIMIZED,
                        "options": cls.ATTENTION_NAMES,
                    },
                ),
            }
        }

    RETURN_TYPES = (IO.MODEL,)
    FUNCTION = "patch"

    CATEGORY = "advanced/model"
    EXPERIMENTAL = True
    DESCRIPTION = "Replaces diffusion model's attention with another registered attention function"

    ATTENTION_OPTIMIZED = "optimized"
    ATTENTION_NAMES = [ATTENTION_OPTIMIZED, *attn.REGISTERED_ATTENTION_FUNCTIONS.keys()]

    def patch(self, model: ModelPatcher, attention: str):
        m = model.clone()

        attention_function: Callable = attn.get_attention_function(attention)  # type: ignore

        def attention_override(_, *args, **kwargs):
            return attention_function(*args, **kwargs)

        options = m.model_options["transformer_options"].copy()
        options["optimized_attention_override"] = attention_override
        m.model_options["transformer_options"] = options

        return (m,)


class CLIPAttentionSelector(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "clip": (IO.CLIP, {}),
                "attention": (
                    IO.COMBO,
                    {
                        "default": cls.ATTENTION_OPTIMIZED,
                        "options": cls.ATTENTION_NAMES,
                    },
                ),
            }
        }

    RETURN_TYPES = (IO.CLIP,)
    FUNCTION = "patch"

    CATEGORY = "advanced/model"
    EXPERIMENTAL = True
    DESCRIPTION = "Replaces text model's attention with another registered attention function"

    ATTENTION_OPTIMIZED = "optimized"
    ATTENTION_NAMES = [ATTENTION_OPTIMIZED, *attn.REGISTERED_ATTENTION_FUNCTIONS.keys()]

    def patch(self, clip: CLIP, attention: str):
        c = clip.clone()
        patcher = c.patcher

        attention_function: Callable = attn.get_attention_function(attention)  # type: ignore

        def attention_override(_, *args, **kwargs):
            return attention_function(*args, **kwargs)

        options = patcher.model_options["transformer_options"].copy()
        options["optimized_attention_override"] = attention_override
        patcher.model_options["transformer_options"] = options

        return (c,)


NODE_CLASS_MAPPINGS = {
    "ModelAttentionSelector": ModelAttentionSelector,
    "CLIPAttentionSelector": CLIPAttentionSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelAttentionSelector": "Model Attention Selector",
    "CLIPAttentionSelector": "CLIP Attention Selector",
}
