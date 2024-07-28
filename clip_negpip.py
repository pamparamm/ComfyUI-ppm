# Original implementation by laksjdjf and hako-mikan licensed under AGPL-3.0
# https://github.com/laksjdjf/cd-tuner_negpip-ComfyUI/blob/938b838546cf774dc8841000996552cef52cccf3/negpip.py#L43-L84
# https://github.com/hako-mikan/sd-webui-negpip
from functools import partial
import torch

from comfy import model_management
from comfy.model_patcher import ModelPatcher
from comfy.sd import CLIP
from comfy.sd1_clip import SD1ClipModel, gen_empty_tokens, ClipTokenWeightEncoder
from comfy.sdxl_clip import SDXLClipModel, SDXLRefinerClipModel


def has_negpip(model_options: dict):
    try:
        return _negpip_attn.__class__ in map(lambda _: _.__class__, model_options["transformer_options"]["patches"]["attn2_patch"])
    except KeyError:
        return False


def _negpip_attn(q, k, v, extra_options):
    new_k = k[:, 0::2]
    new_v = v[:, 1::2]
    return q, new_k, new_v


def _encode_token_weights_negpip(_self: ClipTokenWeightEncoder, token_weight_pairs):
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
        to_encode.append(gen_empty_tokens(_self.special_tokens, max_token_len))

    out, pooled = _self.encode(to_encode)
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
        return out[-1:].to(model_management.intermediate_device()), first_pooled
    return torch.cat(output, dim=-2).to(model_management.intermediate_device()), first_pooled


class CLIPNegPip:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "patch"

    CATEGORY = "conditioning"

    def patch(self, model: ModelPatcher, clip: CLIP):
        m = model.clone()
        c = clip.clone()

        clip_model = type(c.cond_stage_model)
        valid_models = [SD1ClipModel, SDXLClipModel, SDXLRefinerClipModel]
        is_clip_patched = False

        if clip_model in valid_models and not has_negpip(m.model_options):
            if hasattr(c.patcher.model, "clip_g"):
                c.patcher.add_object_patch("clip_g.encode_token_weights", partial(_encode_token_weights_negpip, c.patcher.model.clip_g))
                is_clip_patched = True
            if hasattr(c.patcher.model, "clip_l"):
                c.patcher.add_object_patch("clip_l.encode_token_weights", partial(_encode_token_weights_negpip, c.patcher.model.clip_l))
                is_clip_patched = True
            if is_clip_patched:
                m.set_model_attn2_patch(_negpip_attn)

        return (m, c)
