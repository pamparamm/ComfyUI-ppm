import itertools
from typing import Any

import torch
from torch.types import Device

from .common import COND, UNCOND, CondLike, lcm_for_list, reshape_mask

COND_UNCOND_COUPLE_KEY = "cond_or_uncond_couple"


def _split_kv_cond(cond: torch.Tensor, has_negpip: bool) -> tuple[torch.Tensor, torch.Tensor]:
    if not has_negpip:
        return (cond, cond)

    cond_k, cond_v = cond[:, 0::2], cond[:, 1::2]
    # Prevent attention errors on shape mismatch
    return (cond_k, cond_v) if cond_k.shape == cond_v.shape else (cond, cond)


def unet_attn2_couple_wrapper(
    base_cond: CondLike,
    cond_inputs: list[CondLike],
    num_conds: int,
    has_negpip: bool,
    device: Device,
    dtype: torch.dtype,
):
    conds: list[torch.Tensor] = [cond[0][0].to(device, dtype=dtype) for cond in cond_inputs]

    base_strength: float = base_cond[0][1].get("strength", 1.0)
    strengths: list[float] = [cond[0][1].get("strength", 1.0) for cond in cond_inputs]

    conds_kv = [_split_kv_cond(cond, has_negpip) for cond in conds]

    num_tokens_k = [cond[0].shape[1] for cond in conds_kv]
    num_tokens_v = [cond[1].shape[1] for cond in conds_kv]

    def attn2_patch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, extra_options):
        cond_or_uncond = extra_options["cond_or_uncond"]
        cond_or_uncond_couple = extra_options[COND_UNCOND_COUPLE_KEY] = list(cond_or_uncond)

        num_chunks = len(cond_or_uncond)
        bs = q.shape[0] // num_chunks

        if len(conds_kv) > 0:
            q_chunks = q.chunk(num_chunks, dim=0)
            k_chunks = k.chunk(num_chunks, dim=0)
            v_chunks = v.chunk(num_chunks, dim=0)
            lcm_tokens_k = lcm_for_list(num_tokens_k + [k.shape[1]])
            lcm_tokens_v = lcm_for_list(num_tokens_v + [v.shape[1]])
            conds_k_tensor = torch.cat(
                [
                    cond[0].repeat(bs, lcm_tokens_k // num_tokens_k[i], 1) * strengths[i]
                    for i, cond in enumerate(conds_kv)
                ],
                dim=0,
            )
            conds_v_tensor = torch.cat(
                [
                    cond[1].repeat(bs, lcm_tokens_v // num_tokens_v[i], 1) * strengths[i]
                    for i, cond in enumerate(conds_kv)
                ],
                dim=0,
            )

            qs, ks, vs = [], [], []
            cond_or_uncond_couple.clear()
            for i, cond_type in enumerate(cond_or_uncond):
                q_target = q_chunks[i]
                k_target = k_chunks[i].repeat(1, lcm_tokens_k // k.shape[1], 1)
                v_target = v_chunks[i].repeat(1, lcm_tokens_v // v.shape[1], 1)
                if cond_type == UNCOND:
                    qs.append(q_target)
                    ks.append(k_target)
                    vs.append(v_target)
                    cond_or_uncond_couple.append(UNCOND)
                else:
                    qs.append(q_target.repeat(num_conds, 1, 1))
                    ks.append(torch.cat([k_target * base_strength, conds_k_tensor], dim=0))
                    vs.append(torch.cat([v_target * base_strength, conds_v_tensor], dim=0))
                    cond_or_uncond_couple.extend(itertools.repeat(COND, num_conds))

            qs = torch.cat(qs, dim=0)
            ks = torch.cat(ks, dim=0)
            vs = torch.cat(vs, dim=0)

            return qs, ks, vs

        return q, k, v

    return attn2_patch


def unet_attn2_output_couple_wrapper(mask: torch.Tensor):
    def attn2_output_patch(out: torch.Tensor, extra_options: dict[str, Any]):
        cond_or_uncond = extra_options[COND_UNCOND_COUPLE_KEY]
        size = tuple(extra_options["activations_shape"][-2:])
        bs = out.shape[0] // len(cond_or_uncond)
        num_tokens = out.shape[1]
        mask_downsample = reshape_mask(mask, size, bs, num_tokens)

        outputs = []
        cond_outputs = []
        i_cond = 0
        for i, cond_type in enumerate(cond_or_uncond):
            pos, next_pos = i * bs, (i + 1) * bs

            if cond_type == UNCOND:
                outputs.append(out[pos:next_pos])
            else:
                pos_cond, next_pos_cond = i_cond * bs, (i_cond + 1) * bs
                masked_output = out[pos:next_pos] * mask_downsample[pos_cond:next_pos_cond]
                cond_outputs.append(masked_output)
                i_cond += 1

        if len(cond_outputs) > 0:
            cond_output = torch.stack(cond_outputs).sum(0)
            outputs.append(cond_output)

        return torch.cat(outputs, dim=0)

    return attn2_output_patch
