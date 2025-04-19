# Original implementation by laksjdjf, hako-mikan, Haoming02 licensed under GPL-3.0
# https://github.com/laksjdjf/cgem156-ComfyUI/blob/1f5533f7f31345bafe4b833cbee15a3c4ad74167/scripts/attention_couple/node.py
# https://github.com/Haoming02/sd-forge-couple/blob/e8e258e982a8d149ba59a4bc43b945467604311c/scripts/attention_couple.py
import torch
import torch.nn.functional as F
import math
import itertools
import comfy.model_management
from comfy.model_patcher import ModelPatcher

from .clip_negpip import has_negpip

COND = 0
UNCOND = 1


# Naive and totally inaccurate way to factorize target_res into rescaled integer width/height
def rescale_size(
    width: int,
    height: int,
    target_res: int,
    *,
    tolerance=2,
) -> tuple[int, int]:
    tolerance = min(target_res, tolerance)

    def get_neighbors(num: float):
        if num < 1:
            return None
        numi = int(num)
        return tuple(
            numi + adj
            for adj in sorted(
                range(
                    -min(numi - 1, tolerance),
                    tolerance + 1 + math.ceil(num - numi),
                ),
                key=abs,
            )
        )

    scale = math.sqrt(height * width / target_res)
    height_scaled, width_scaled = height / scale, width / scale
    height_rounded = get_neighbors(height_scaled)
    width_rounded = get_neighbors(width_scaled)
    for h, w in itertools.zip_longest(height_rounded, width_rounded):  # type: ignore
        h_adj = target_res / w if w is not None else 0.1
        if h_adj % 1 == 0:
            return (w, int(h_adj))
        if h is None:
            continue
        w_adj = target_res / h
        if w_adj % 1 == 0:
            return (int(w_adj), h)
    msg = f"Can't rescale {width} and {height} to fit {target_res}"
    raise ValueError(msg)


def get_mask(mask, batch_size, num_tokens, extra_options):
    if "activations_shape" in extra_options:
        activations_shape = extra_options["activations_shape"]
        size = (activations_shape[3], activations_shape[2])
    else:
        original_shape = extra_options["original_shape"]
        image_width: int = original_shape[3]
        image_height: int = original_shape[2]
        size = rescale_size(image_width, image_height, num_tokens)

    num_conds = mask.shape[0]
    mask_downsample = F.interpolate(mask, size=size, mode="nearest")
    mask_downsample_reshaped = mask_downsample.view(num_conds, num_tokens, 1).repeat_interleave(batch_size, dim=0)

    return mask_downsample_reshaped


def lcm(a, b):
    return a * b // math.gcd(a, b)


def lcm_for_list(numbers):
    current_lcm = numbers[0]
    for number in numbers[1:]:
        current_lcm = lcm(current_lcm, number)
    return current_lcm


class AttentionCouplePPM:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "base_mask": ("MASK",),
            },
            "optional": {
                "cond_1": ("CONDITIONING",),
                "mask_1": ("MASK",),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "advanced/model"

    def patch(self, model: ModelPatcher, base_mask, **kwargs: dict):
        self.batch_size = 0

        m = model.clone()
        dtype = m.model.diffusion_model.dtype
        device = comfy.model_management.get_torch_device()

        num_conds = len(kwargs) // 2 + 1
        original_cond_or_uncond = [1, 0]

        mask = [base_mask] + [kwargs[f"mask_{i}"] for i in range(1, num_conds)]
        mask = torch.stack(mask, dim=0).to(device, dtype=dtype)
        if mask.sum(dim=0).min() <= 0:
            raise ValueError("Masks contain non-filled areas")
        mask = mask / mask.sum(dim=0, keepdim=True)

        conds: list[torch.Tensor] = [kwargs[f"cond_{i}"][0][0].to(device, dtype=dtype) for i in range(1, num_conds)]

        if has_negpip(m.model_options):
            conds_kv = [(cond[:, 0::2], cond[:, 1::2]) for cond in conds]
        else:
            conds_kv = [(cond, cond) for cond in conds]

        def attn2_patch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, extra_options):
            nonlocal original_cond_or_uncond
            original_cond_or_uncond = extra_options["cond_or_uncond"]

            cond_or_uncond = original_cond_or_uncond
            num_chunks = len(cond_or_uncond)
            self.batch_size = q.shape[0] // num_chunks
            if len(conds_kv) > 0:
                q_chunks = q.chunk(num_chunks, dim=0)
                k_chunks = k.chunk(num_chunks, dim=0)
                v_chunks = v.chunk(num_chunks, dim=0)
                num_tokens_k = [cond[0].shape[1] for cond in conds_kv]
                num_tokens_v = [cond[1].shape[1] for cond in conds_kv]
                lcm_tokens_k = lcm_for_list(num_tokens_k + [k.shape[1]])
                lcm_tokens_v = lcm_for_list(num_tokens_v + [v.shape[1]])
                conds_k_tensor = torch.cat(
                    [
                        cond[0].repeat(self.batch_size, lcm_tokens_k // num_tokens_k[i], 1)
                        for i, cond in enumerate(conds_kv)
                    ],
                    dim=0,
                )
                conds_v_tensor = torch.cat(
                    [
                        cond[1].repeat(self.batch_size, lcm_tokens_v // num_tokens_v[i], 1)
                        for i, cond in enumerate(conds_kv)
                    ],
                    dim=0,
                )

                qs, ks, vs = [], [], []
                new_cond_or_uncond = []
                for i, cond_type in enumerate(cond_or_uncond):
                    q_target = q_chunks[i]
                    k_target = k_chunks[i].repeat(1, lcm_tokens_k // k.shape[1], 1)
                    v_target = v_chunks[i].repeat(1, lcm_tokens_v // v.shape[1], 1)
                    if cond_type == UNCOND:
                        qs.append(q_target)
                        ks.append(k_target)
                        vs.append(v_target)
                        new_cond_or_uncond.append(UNCOND)
                    else:
                        qs.append(q_target.repeat(num_conds, 1, 1))
                        ks.append(torch.cat([k_target, conds_k_tensor], dim=0))
                        vs.append(torch.cat([v_target, conds_v_tensor], dim=0))
                        new_cond_or_uncond.extend(itertools.repeat(COND, num_conds))

                qs = torch.cat(qs, dim=0)
                ks = torch.cat(ks, dim=0)
                vs = torch.cat(vs, dim=0)

                extra_options["cond_or_uncond"] = new_cond_or_uncond

                return qs, ks, vs

            return q, k, v

        def attn2_output_patch(out, extra_options):
            cond_or_uncond = extra_options["cond_or_uncond"]
            bs = self.batch_size
            mask_downsample = get_mask(mask, self.batch_size, out.shape[1], extra_options)
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

            cond_output = torch.stack(cond_outputs).sum(0)
            outputs.append(cond_output)
            extra_options["cond_or_uncond"] = original_cond_or_uncond
            return torch.cat(outputs, dim=0)

        m.set_model_attn2_patch(attn2_patch)
        m.set_model_attn2_output_patch(attn2_output_patch)

        return (m,)
