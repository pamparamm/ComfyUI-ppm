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
    for h, w in itertools.zip_longest(height_rounded, width_rounded):
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


def get_mask(mask, batch_size, num_tokens, original_shape):
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
            cond_or_unconds = extra_options["cond_or_uncond"]
            num_chunks = len(cond_or_unconds)
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
                    [cond[0].repeat(self.batch_size, lcm_tokens_k // num_tokens_k[i], 1) for i, cond in enumerate(conds_kv)],
                    dim=0,
                )
                conds_v_tensor = torch.cat(
                    [cond[1].repeat(self.batch_size, lcm_tokens_v // num_tokens_v[i], 1) for i, cond in enumerate(conds_kv)],
                    dim=0,
                )

                qs, ks, vs = [], [], []
                for i, cond_or_uncond in enumerate(cond_or_unconds):
                    q_target = q_chunks[i]
                    k_target = k_chunks[i].repeat(1, lcm_tokens_k // k.shape[1], 1)
                    v_target = v_chunks[i].repeat(1, lcm_tokens_v // v.shape[1], 1)
                    if cond_or_uncond == UNCOND:
                        qs.append(q_target)
                        ks.append(k_target)
                        vs.append(v_target)
                    else:
                        qs.append(q_target.repeat(num_conds, 1, 1))
                        ks.append(torch.cat([k_target, conds_k_tensor], dim=0))
                        vs.append(torch.cat([v_target, conds_v_tensor], dim=0))

                qs = torch.cat(qs, dim=0)
                ks = torch.cat(ks, dim=0)
                vs = torch.cat(vs, dim=0)

                return qs, ks, vs

            return q, k, v

        def attn2_output_patch(out, extra_options):
            cond_or_unconds = extra_options["cond_or_uncond"]
            mask_downsample = get_mask(mask, self.batch_size, out.shape[1], extra_options["original_shape"])
            outputs = []
            pos = 0
            for cond_or_uncond in cond_or_unconds:
                if cond_or_uncond == UNCOND:
                    outputs.append(out[pos : pos + self.batch_size])
                    pos += self.batch_size
                else:
                    masked_output = (out[pos : pos + num_conds * self.batch_size] * mask_downsample).view(
                        num_conds, self.batch_size, out.shape[1], out.shape[2]
                    )
                    masked_output = masked_output.sum(dim=0)
                    outputs.append(masked_output)
                    pos += num_conds * self.batch_size
            return torch.cat(outputs, dim=0)

        m.set_model_attn2_patch(attn2_patch)
        m.set_model_attn2_output_patch(attn2_output_patch)

        return (m,)
