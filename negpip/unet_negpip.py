from typing import Any

import torch

from comfy import model_management
from comfy.sd1_clip import SDClipModel, gen_empty_tokens


def sdxl_attn2_negpip(q, k, v, extra_options: dict[str, Any]):
    new_k = k[:, 0::2]
    new_v = v[:, 1::2]
    return q, new_k, new_v


# Modified version of ClipTokenWeightEncoder.encode_token_weights
def encode_token_weights_negpip(
    self: SDClipModel,
    token_weight_pairs,
):
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
        first_pooled = pooled[0:1].to(device=model_management.intermediate_device())
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
                        zk[i][j] = (zk[i][j] - z_empty[j]) * abs(weight) + z_empty[j]
                        zv[i][j] = (zv[i][j] - z_empty[j]) * abs(weight) + z_empty[j]
                        if weight < 0:
                            zv[i][j] = -zv[i][j]

        z = torch.zeros_like(zk).repeat(1, 2, 1)
        for i in range(zk.shape[1]):
            z[:, 2 * i, :] = zk[:, i, :]
            z[:, 2 * i + 1, :] = zv[:, i, :]
        output.append(z)

    if len(output) == 0:
        r = (out[-1:].to(device=model_management.intermediate_device()), first_pooled)
    else:
        r = (torch.cat(output, dim=-2).to(device=model_management.intermediate_device()), first_pooled)

    if len(o) > 2:
        extra = {}
        for k in o[2]:
            v = o[2][k]
            if k == "attention_mask":
                v = v[:sections].flatten().unsqueeze(dim=0).to(device=model_management.intermediate_device())
            extra[k] = v

        r = r + (extra,)
    return r
