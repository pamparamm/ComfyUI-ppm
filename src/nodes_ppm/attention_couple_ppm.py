# Original implementation by laksjdjf, hako-mikan, Haoming02 licensed under GPL-3.0
# https://github.com/laksjdjf/cgem156-ComfyUI/blob/1f5533f7f31345bafe4b833cbee15a3c4ad74167/scripts/attention_couple/node.py
# https://github.com/Haoming02/sd-forge-couple/blob/e8e258e982a8d149ba59a4bc43b945467604311c/scripts/attention_couple.py
import torch

import comfy.model_management
import comfy.patcher_extension
from comfy.ldm.anima.model import Anima as AnimaDIT
from comfy.ldm.cosmos.predict2 import Attention as CosmosAttention
from comfy.model_base import SDXL, Anima, BaseModel, SDXLRefiner
from comfy.model_patcher import ModelPatcher
from comfy_api.latest import io

from ..attention_couple.anima_couple import (
    anima_couple_sample_wrapper,
    anima_forward_wrapper,
    cosmos_attention_forward_couple,
)
from ..attention_couple.common import CondLike
from ..attention_couple.unet_couple import unet_attn2_couple_wrapper, unet_attn2_output_couple_wrapper
from .clip_negpip import has_negpip

COUPLE_WRAPPER_KEY = "ppm_attention_couple"


# TODO Migrate to io.Autogrow.TemplatePrefix
class AttentionCouplePPM(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AttentionCouplePPM",
            display_name="Attention Couple (PPM)",
            category="advanced/model",
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input(
                    "base_cond",
                    tooltip="Positive conditioning from KSampler/SamplerCustom node.\n"
                    "Can be optionally scaled up/down by using ConditioningSetAreaStrength node.",
                ),
                io.Mask.Input("base_mask"),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        model: ModelPatcher = kwargs["model"]
        base_cond: CondLike = kwargs["base_cond"]
        base_mask = kwargs["base_mask"]

        m = model.clone()
        dtype = m.model.diffusion_model.dtype
        device = comfy.model_management.get_torch_device()
        _has_negpip = has_negpip(m.model_options)

        num_conds = len([k for k in kwargs.keys() if "cond" in k])
        cond_inputs: list[CondLike] = [kwargs[f"cond_{i}"] for i in range(1, num_conds)]
        mask_inputs: list[torch.Tensor] = [kwargs[f"mask_{i}"] for i in range(1, num_conds)]

        mask = [base_mask] + mask_inputs
        mask = torch.stack(mask, dim=0).to(device, dtype=dtype)
        if mask.sum(dim=0).min() <= 0:
            raise ValueError("Masks contain non-filled areas")
        mask = mask / mask.sum(dim=0, keepdim=True)

        model_type = type(m.model)
        diffusion_model = m.get_model_object("diffusion_model")

        # SD1.* and SDXL
        if issubclass(model_type, SDXL) or issubclass(model_type, SDXLRefiner) or model_type == BaseModel:
            m.set_model_attn2_patch(
                unet_attn2_couple_wrapper(
                    base_cond,
                    cond_inputs,
                    num_conds,
                    _has_negpip,
                    device,
                    dtype,
                )
            )
            m.set_model_attn2_output_patch(unet_attn2_output_couple_wrapper(mask))

        if issubclass(model_type, Anima):
            anima_model: AnimaDIT = diffusion_model  # type: ignore
            m.add_wrapper_with_key(
                comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE,
                COUPLE_WRAPPER_KEY,
                anima_couple_sample_wrapper(cond_inputs, device),
            )
            m.add_wrapper_with_key(
                comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                COUPLE_WRAPPER_KEY,
                anima_forward_wrapper,
            )

            for block_name, block in (
                (n, b) for n, b in anima_model.named_modules() if "cross_attn" in n and isinstance(b, CosmosAttention)
            ):
                attn_forward_prev = m.get_model_object(f"diffusion_model.{block_name}.forward")
                m.add_object_patch(
                    f"diffusion_model.{block_name}.forward",
                    cosmos_attention_forward_couple(attn_forward_prev, mask, num_conds),
                )

        return io.NodeOutput(m)


NODES = [AttentionCouplePPM]
