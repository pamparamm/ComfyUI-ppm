# Original implementation by laksjdjf and hako-mikan licensed under AGPL-3.0
# https://github.com/laksjdjf/cd-tuner_negpip-ComfyUI/blob/938b838546cf774dc8841000996552cef52cccf3/negpip.py#L43-L84
# https://github.com/hako-mikan/sd-webui-negpip
from functools import partial
from typing import Any

import torch

from comfy import model_management
from comfy.ldm.anima.model import Anima as AnimaDIT
from comfy.ldm.cosmos.predict2 import Attention as CosmosAttention
from comfy.model_base import SDXL, Anima, BaseModel, Flux, SDXLRefiner
from comfy.model_patcher import ModelPatcher
from comfy.sd import CLIP
from comfy.sd1_clip import SDClipModel, gen_empty_tokens
from comfy_api.latest import io

from ..compat.advanced_encode import patch_adv_encode
from ..dit.anima_negpip import (
    anima_extra_conds_negpip,
    anima_forward_negpip,
    cosmos_attention_forward_negpip,
)
from ..dit.flux_negpip import flux_forward_orig_negpip

NEGPIP_OPTION = "ppm_negpip"
SUPPORTED_ENCODERS = [
    "clip_g",
    "clip_l",
    "t5xxl",
    "llama",
    "qwen3_06b",
]


def has_negpip(model_options: dict):
    return model_options.get(NEGPIP_OPTION, False)


def negpip_attn(q, k, v, extra_options):
    new_k = k[:, 0::2]
    new_v = v[:, 1::2]
    return q, new_k, new_v


def encode_token_weights_negpip(self: SDClipModel, token_weight_pairs):
    to_encode = list()
    max_token_len = 0
    has_weights = False
    for x in token_weight_pairs:
        tokens = list(map(lambda a: a[0], x))
        max_token_len = max(len(tokens), max_token_len)
        has_weights = has_weights or not all(map(lambda a: a[1] == 1.0, x))
        to_encode.append(tokens)

    sections = len(to_encode)
    if has_weights or sections == 0:
        if hasattr(self, "gen_empty_tokens"):
            to_encode.append(self.gen_empty_tokens(self.special_tokens, max_token_len))  # type: ignore
        else:
            to_encode.append(gen_empty_tokens(self.special_tokens, max_token_len))

    o = self.encode(to_encode)
    out, pooled = o[:2]

    if pooled is not None:
        first_pooled = pooled[0:1].to(model_management.intermediate_device())
    else:
        first_pooled = pooled

    output = []
    for k in range(0, sections):
        zk = out[k : k + 1].clone()
        zv = out[k : k + 1].clone()
        if has_weights:
            z_empty = out[-1]
            for i in range(len(zk)):
                for j in range(len(zk[i])):
                    weight = token_weight_pairs[k][j][1]
                    if weight != 1.0:
                        if weight < 0:
                            weight = -weight
                            sign = -1
                        else:
                            sign = 1
                        zk[i][j] = (zk[i][j] - z_empty[j]) * weight + z_empty[j]
                        zv[i][j] = sign * ((zv[i][j] - z_empty[j]) * weight + z_empty[j])

        z = torch.zeros_like(zk).repeat(1, 2, 1)
        for i in range(zk.shape[1]):
            z[:, 2 * i, :] += zk[:, i, :]
            z[:, 2 * i + 1, :] += zv[:, i, :]
        output.append(z)

    if len(output) == 0:
        r = (out[-1:].to(model_management.intermediate_device()), first_pooled)
    else:
        r = (torch.cat(output, dim=-2).to(model_management.intermediate_device()), first_pooled)

    if len(o) > 2:
        extra = {}
        for k in o[2]:
            v = o[2][k]
            if k == "attention_mask":
                v = v[:sections].flatten().unsqueeze(dim=0).to(model_management.intermediate_device())
            extra[k] = v

        r = r + (extra,)
    return r


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
        diffusion_model = m.model.diffusion_model

        # SD1.* and SDXL
        if issubclass(model_type, SDXL) or issubclass(model_type, SDXLRefiner) or model_type == BaseModel:
            for encoder in encoders:
                c.patcher.add_object_patch(
                    f"{encoder}.encode_token_weights",
                    partial(encode_token_weights_negpip, getattr(c.patcher.model, encoder)),
                )
            m.set_model_attn2_patch(negpip_attn)
            return True

        # Flux (probably broken)
        if issubclass(model_type, Flux):
            for encoder in encoders:
                c.patcher.add_object_patch(
                    f"{encoder}.encode_token_weights",
                    partial(encode_token_weights_negpip, getattr(c.patcher.model, encoder)),
                )
            m.add_object_patch("diffusion_model.forward_orig", partial(flux_forward_orig_negpip, diffusion_model))
            return True

        # Anima
        if issubclass(model_type, Anima):
            diffusion_model: AnimaDIT
            m.add_object_patch(
                "extra_conds",
                partial(anima_extra_conds_negpip, m.model),
            )
            m.add_object_patch(
                "diffusion_model._forward",
                partial(anima_forward_negpip, diffusion_model._forward),
            )

            for block_name, block in (
                (n, b)
                for n, b in diffusion_model.named_modules()
                if "cross_attn" in n and isinstance(b, CosmosAttention)
            ):
                m.add_object_patch(
                    f"diffusion_model.{block_name}.forward", partial(cosmos_attention_forward_negpip, block)
                )

            return True

        return False


NODES = [CLIPNegPip]
