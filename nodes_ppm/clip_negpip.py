# Original implementation by laksjdjf and hako-mikan licensed under AGPL-3.0
# https://github.com/laksjdjf/cd-tuner_negpip-ComfyUI/blob/938b838546cf774dc8841000996552cef52cccf3/negpip.py#L43-L84
# https://github.com/hako-mikan/sd-webui-negpip
from functools import partial
from typing import Any

import torch
from comfy import model_management
from comfy.comfy_types.node_typing import IO, ComfyNodeABC, InputTypeDict
from comfy.model_base import Flux, HunyuanVideo
from comfy.model_patcher import ModelPatcher
from comfy.sd import CLIP
from comfy.sd1_clip import SDClipModel, gen_empty_tokens

from ..compat.advanced_encode import patch_adv_encode
from ..dit.flux_negpip import flux_forward_orig_negpip
from ..dit.hunyuan_video_negpip import (
    hunyuan_video_clip_encode_token_weights_negpip,
    hunyuan_video_forward_orig_negpip,
)

NEGPIP_OPTION = "ppm_negpip"


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


class CLIPNegPip(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "model": (IO.MODEL, {}),
                "clip": (IO.CLIP, {}),
            }
        }

    RETURN_TYPES = (IO.MODEL, IO.CLIP)
    FUNCTION = "patch"

    CATEGORY = "conditioning"

    def patch(self, model: ModelPatcher, clip: CLIP):
        m = model.clone()
        c = clip.clone()
        model_options: dict[str, Any] = m.model_options
        clip_options: dict[str, Any] = clip.patcher.model_options

        patch_adv_encode()

        diffusion_model = type(m.model)
        is_clip_patched = False

        if not has_negpip(model_options):
            if hasattr(c.patcher.model, "clip_g"):
                c.patcher.add_object_patch(
                    "clip_g.encode_token_weights", partial(encode_token_weights_negpip, c.patcher.model.clip_g)
                )
                is_clip_patched = True
            if hasattr(c.patcher.model, "clip_l"):
                c.patcher.add_object_patch(
                    "clip_l.encode_token_weights", partial(encode_token_weights_negpip, c.patcher.model.clip_l)
                )
                is_clip_patched = True
            if hasattr(c.patcher.model, "t5xxl"):
                c.patcher.add_object_patch(
                    "t5xxl.encode_token_weights", partial(encode_token_weights_negpip, c.patcher.model.t5xxl)
                )
                is_clip_patched = True
            if hasattr(c.patcher.model, "llama"):
                c.patcher.add_object_patch(
                    "llama.encode_token_weights", partial(encode_token_weights_negpip, c.patcher.model.llama)
                )
                is_clip_patched = True

            if is_clip_patched:
                m.set_model_attn2_patch(negpip_attn)
                model_options[NEGPIP_OPTION] = True
                clip_options[NEGPIP_OPTION] = True

                if issubclass(diffusion_model, Flux):
                    m.add_object_patch(
                        "diffusion_model.forward_orig", partial(flux_forward_orig_negpip, m.model.diffusion_model)
                    )
                if issubclass(diffusion_model, HunyuanVideo):
                    c.patcher.add_object_patch(
                        "encode_token_weights",
                        partial(hunyuan_video_clip_encode_token_weights_negpip, c.patcher.model),
                    )
                    m.add_object_patch(
                        "diffusion_model.forward_orig",
                        partial(hunyuan_video_forward_orig_negpip, m.model.diffusion_model),
                    )

        return (m, c)
