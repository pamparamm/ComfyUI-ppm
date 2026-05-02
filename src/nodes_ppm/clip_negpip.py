# Original implementation by laksjdjf and hako-mikan licensed under AGPL-3.0
# https://github.com/laksjdjf/cd-tuner_negpip-ComfyUI/blob/938b838546cf774dc8841000996552cef52cccf3/negpip.py#L43-L84
# https://github.com/hako-mikan/sd-webui-negpip
from functools import partial
from typing import Any

import comfy.patcher_extension
from comfy.ldm.anima.model import Anima as AnimaDIT
from comfy.ldm.cosmos.predict2 import Attention as CosmosAttention
from comfy.ldm.flux.model import Flux as FluxDIT
from comfy.model_base import SDXL, Anima, BaseModel, Flux, SDXLRefiner
from comfy.model_patcher import ModelPatcher
from comfy.sd import CLIP
from comfy_api.latest import io

from ..compat.advanced_encode import patch_adv_encode
from ..negpip.anima_negpip import (
    anima_extra_conds_negpip,
    cosmos_attention_forward_negpip,
    cosmos_diffusion_negpip_wrapper,
)
from ..negpip.flux_negpip import flux_forward_orig_negpip
from ..negpip.unet_negpip import (
    encode_token_weights_negpip,
    sdxl_attn2_negpip,
)

NEGPIP_OPTION = "ppm_negpip"
SUPPORTED_ENCODERS = [
    "clip_g",
    "clip_l",
    "t5xxl",
    "llama",
    "qwen3_06b",
]


def has_negpip(model_options: dict) -> bool:
    return model_options.get(NEGPIP_OPTION, False)


class CLIPNegPip(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CLIPNegPip",
            display_name="CLIP NegPip",
            category="conditioning",
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
            ],
            outputs=[
                io.Model.Output(),
                io.Clip.Output(),
            ],
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        model: ModelPatcher = kwargs["model"]
        clip: CLIP = kwargs["clip"]
        m = model.clone()
        c = clip.clone()
        model_options: dict[str, Any] = m.model_options
        clip_options: dict[str, Any] = c.patcher.model_options

        encoders = [e for e in SUPPORTED_ENCODERS if hasattr(c.patcher.model, e)]
        if len(encoders) == 0:
            return io.NodeOutput(m, c)

        if not has_negpip(model_options):
            patch_adv_encode()
            is_patched = cls.patch_negpip(m, c, encoders)

            if is_patched:
                model_options[NEGPIP_OPTION] = True
                clip_options[NEGPIP_OPTION] = True

        return io.NodeOutput(m, c)

    @staticmethod
    def patch_negpip(m: ModelPatcher, c: CLIP, encoders: list[str]):
        model_type = type(m.model)
        diffusion_model = m.get_model_object("diffusion_model")

        # SD1.* and SDXL
        if issubclass(model_type, SDXL) or issubclass(model_type, SDXLRefiner) or model_type == BaseModel:
            for encoder in encoders:
                c.patcher.add_object_patch(
                    f"{encoder}.encode_token_weights",
                    partial(encode_token_weights_negpip, getattr(c.patcher.model, encoder)),
                )
            m.set_model_attn2_patch(sdxl_attn2_negpip)
            return True

        # Flux (probably broken)
        if issubclass(model_type, Flux):
            flux_model: FluxDIT = diffusion_model  # type: ignore
            for encoder in encoders:
                c.patcher.add_object_patch(
                    f"{encoder}.encode_token_weights",
                    partial(encode_token_weights_negpip, getattr(c.patcher.model, encoder)),
                )
            m.add_object_patch("diffusion_model.forward_orig", partial(flux_forward_orig_negpip, flux_model))
            return True

        # Anima
        if issubclass(model_type, Anima):
            anima_model: AnimaDIT = diffusion_model  # type: ignore
            m.add_object_patch(
                "extra_conds",
                partial(anima_extra_conds_negpip, m.model.extra_conds),
            )
            m.add_wrapper_with_key(
                comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                NEGPIP_OPTION,
                cosmos_diffusion_negpip_wrapper,
            )

            for block_name, block in (
                (n, b) for n, b in anima_model.named_modules() if "cross_attn" in n and isinstance(b, CosmosAttention)
            ):
                m.add_object_patch(
                    f"diffusion_model.{block_name}.forward", partial(cosmos_attention_forward_negpip, block)
                )

            return True

        return False


NODES = [CLIPNegPip]
