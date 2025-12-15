# Original implementation by laksjdjf, hako-mikan, Haoming02 licensed under GPL-3.0
# https://github.com/laksjdjf/cgem156-ComfyUI/blob/1f5533f7f31345bafe4b833cbee15a3c4ad74167/scripts/attention_couple/node.py
# https://github.com/Haoming02/sd-forge-couple/blob/e8e258e982a8d149ba59a4bc43b945467604311c/scripts/attention_couple.py
import itertools
import math

import torch
import torch.nn.functional as F

import comfy.model_management
from comfy.model_patcher import ModelPatcher
from comfy_api.latest import io

from .clip_negpip import has_negpip

COND = 0
UNCOND = 1
COND_UNCOND_COUPLE_OPTION = "cond_or_uncond_couple"


def lcm_for_list(numbers):
    current_lcm = numbers[0]
    for number in numbers[1:]:
        current_lcm = math.lcm(current_lcm, number)
    return current_lcm


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
        base_cond = kwargs["base_cond"]
        base_mask = kwargs["base_mask"]

        m = model.clone()
        dtype = m.model.diffusion_model.dtype
        device = comfy.model_management.get_torch_device()

        num_conds = len(kwargs) // 2 + 1
        cond_inputs: list[list[list]] = [kwargs[f"cond_{i}"] for i in range(1, num_conds)]
        mask_inputs: list[torch.Tensor] = [kwargs[f"mask_{i}"] for i in range(1, num_conds)]

        mask = [base_mask] + mask_inputs
        mask = torch.stack(mask, dim=0).to(device, dtype=dtype)
        if mask.sum(dim=0).min() <= 0:
            raise ValueError("Masks contain non-filled areas")
        mask = mask / mask.sum(dim=0, keepdim=True)

        conds: list[torch.Tensor] = [cond[0][0].to(device, dtype=dtype) for cond in cond_inputs]

        base_strength: float = base_cond[0][1].get("strength", 1.0)
        strengths: list[float] = [cond[0][1].get("strength", 1.0) for cond in cond_inputs]

        _has_negpip = has_negpip(m.model_options)
        conds_kv = [cls.split_kv_cond(cond, _has_negpip) for cond in conds]

        num_tokens_k = [cond[0].shape[1] for cond in conds_kv]
        num_tokens_v = [cond[1].shape[1] for cond in conds_kv]

        def attn2_patch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, extra_options):
            cond_or_uncond = extra_options["cond_or_uncond"]
            cond_or_uncond_couple = extra_options[COND_UNCOND_COUPLE_OPTION] = list(cond_or_uncond)

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

        def attn2_output_patch(out, extra_options):
            cond_or_uncond = extra_options[COND_UNCOND_COUPLE_OPTION]
            size = tuple(extra_options["activations_shape"][-2:])
            bs = out.shape[0] // len(cond_or_uncond)
            num_tokens = out.shape[1]
            mask_downsample = cls.reshape_mask(mask, size, bs, num_tokens)

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

        m.set_model_attn2_patch(attn2_patch)
        m.set_model_attn2_output_patch(attn2_output_patch)

        return io.NodeOutput(m)

    @staticmethod
    def reshape_mask(mask: torch.Tensor, size: tuple[int, int], bs: int, num_tokens: int) -> torch.Tensor:
        num_conds = mask.shape[0]
        mask_downsample = F.interpolate(mask, size=size, mode="nearest")
        mask_downsample_reshaped = mask_downsample.view(num_conds, num_tokens, 1).repeat_interleave(bs, dim=0)

        return mask_downsample_reshaped

    @staticmethod
    def split_kv_cond(cond: torch.Tensor, has_negpip: bool) -> tuple[torch.Tensor, torch.Tensor]:
        if not has_negpip:
            return (cond, cond)

        cond_k, cond_v = cond[:, 0::2], cond[:, 1::2]
        # Prevent attention errors on shape mismatch
        return (cond_k, cond_v) if cond_k.shape == cond_v.shape else (cond, cond)


NODES = [AttentionCouplePPM]
