import itertools
from typing import Any, Callable

import torch
from torch.types import Device

from comfy.ldm.anima.model import Anima as AnimaDIT
from comfy.patcher_extension import WrapperExecutor
from comfy.sampler_helpers import convert_cond
from comfy.samplers import CFGGuider, process_conds

from ..negpip.anima_negpip import COND_NEGPIP_MASK_KEY, NEGPIP_MASK_KEY
from .common import COND, UNCOND, CondLike, lcm_for_list, reshape_mask

CONDS_COUPLE_KEY = "ppm_couple_conds"
COND_UNCOND_COUPLE_KEY = "ppm_couple_cond_or_uncond"
NEGPIP_MASKS_COUPLE_KEY = "ppm_couple_negpip_masks"
NUM_TOKENS_COUPLE_KEY = "ppm_couple_num_tokens"


def cosmos_couple_sample_wrapper(cond_inputs: list[CondLike], device: Device):
    conds_converted = [convert_cond(cond)[0] for cond in cond_inputs]

    def _cosmos_couple_sample_wrapper(executor, *args, **kwargs):
        if len(conds_converted) > 0:
            guider: CFGGuider = args[0]
            extra_options: dict[str, Any] = args[2]
            seed: int = extra_options["seed"]
            noise: torch.Tensor = args[4]
            latent_image: torch.Tensor = args[5]
            denoise_mask: torch.Tensor | None = args[6]
            latent_shapes = [latent_image.shape]

            conds_processed = process_conds(
                guider.inner_model,
                noise,
                {"positive": conds_converted},
                device,
                latent_image,
                denoise_mask,
                seed,
                latent_shapes=latent_shapes,
            )["positive"]

            conds_couple = [cond["model_conds"]["c_crossattn"].cond for cond in conds_processed]
            negpip_masks_couple = [
                cond["model_conds"][COND_NEGPIP_MASK_KEY].cond.to(device)
                for cond in conds_processed
                if COND_NEGPIP_MASK_KEY in cond["model_conds"]
            ]

            model_options: dict[str, Any] = extra_options["model_options"]
            transformer_options: dict[str, Any] = model_options.get("transformer_options", {}).copy()
            transformer_options[CONDS_COUPLE_KEY] = conds_couple
            if len(negpip_masks_couple) > 0:
                transformer_options[NEGPIP_MASKS_COUPLE_KEY] = negpip_masks_couple
            transformer_options[NUM_TOKENS_COUPLE_KEY] = [cond.shape[1] for cond in conds_couple]
            model_options["transformer_options"] = transformer_options

        return executor(*args, **kwargs)

    return _cosmos_couple_sample_wrapper


def cosmos_couple_diffusion_wrapper(mask: torch.Tensor, num_conds: int):
    def _cosmos_couple_diffusion_wrapper(executor: WrapperExecutor, *args, **kwargs):
        anima_model: AnimaDIT = executor.class_obj  # type: ignore

        x: torch.Tensor = args[0]
        transformer_options: dict[str, Any] = kwargs.get("transformer_options", {}).copy()
        patch_spatial = anima_model.patch_spatial

        activations_shape = list(x.shape)
        activations_shape[-2] = activations_shape[-2] // patch_spatial
        activations_shape[-1] = activations_shape[-1] // patch_spatial

        transformer_options["activations_shape"] = activations_shape
        transformer_options["ppm_attn_pre_ca"] = _cosmos_ppm_attn_pre_ca_couple
        transformer_options["ppm_attn_output_ca"] = _cosmos_ppm_attn_output_ca_couple
        kwargs["transformer_options"] = transformer_options

        return executor(*args, **kwargs)

    def _cosmos_ppm_attn_pre_ca_couple(
        transformer_options: dict,
        x: torch.Tensor,
        context: torch.Tensor,
        rope_emb: torch.Tensor,
        _transformer_options,
    ):
        cond_or_uncond_couple = []
        transformer_options = transformer_options.copy()

        if context is None or CONDS_COUPLE_KEY not in transformer_options:
            transformer_options[COND_UNCOND_COUPLE_KEY] = list(transformer_options["cond_or_uncond"])
            return x, context, rope_emb, transformer_options

        c: torch.Tensor = context

        conds: list[torch.Tensor] = transformer_options[CONDS_COUPLE_KEY]
        num_tokens_c: list[int] = transformer_options[NUM_TOKENS_COUPLE_KEY]
        cond_or_uncond = transformer_options["cond_or_uncond"]

        num_chunks = len(cond_or_uncond)
        bs = x.shape[0] // num_chunks

        n: torch.Tensor = transformer_options.get(NEGPIP_MASK_KEY, None)  # type: ignore
        negpip_masks: list[torch.Tensor] = transformer_options.get(NEGPIP_MASKS_COUPLE_KEY, None)  # type: ignore
        has_negpip = n is not None and negpip_masks is not None

        x_chunks = x.chunk(num_chunks, dim=0)
        c_chunks = c.chunk(num_chunks, dim=0)
        lcm_tokens_c = lcm_for_list(num_tokens_c + [c.shape[1]])
        conds_c_tensor = torch.cat(
            [cond.repeat(bs, lcm_tokens_c // num_tokens_c[i], 1) for i, cond in enumerate(conds)],
            dim=0,
        )

        if has_negpip:
            n_chunks = n.chunk(num_chunks, dim=0)
            num_tokens_n: list[int] = [mask.shape[1] for mask in negpip_masks]
            lcm_tokens_n = lcm_for_list(num_tokens_n + [n.shape[1]])
            conds_n_tensor = torch.cat(
                [mask.repeat(bs, lcm_tokens_n // num_tokens_n[i], 1) for i, mask in enumerate(negpip_masks)],
                dim=0,
            )

        xs, cs, ns = [], [], []
        for i, cond_type in enumerate(cond_or_uncond):
            x_target = x_chunks[i]
            c_target = c_chunks[i].repeat(1, lcm_tokens_c // c.shape[1], 1)
            if cond_type == UNCOND:
                xs.append(x_target)
                cs.append(c_target)
                cond_or_uncond_couple.append(UNCOND)
            else:
                xs.append(x_target.repeat(num_conds, 1, 1))
                cs.append(torch.cat([c_target, conds_c_tensor], dim=0))
                cond_or_uncond_couple.extend(itertools.repeat(COND, num_conds))

            if has_negpip:
                n_target = n_chunks[i].repeat(1, lcm_tokens_n // n.shape[1], 1)  # type: ignore
                if cond_type == UNCOND:
                    ns.append(n_target)
                else:
                    ns.append(torch.cat([n_target, conds_n_tensor], dim=0))  # type: ignore

        xs = torch.cat(xs, dim=0)
        cs = torch.cat(cs, dim=0)

        if has_negpip:
            ns = torch.cat(ns, dim=0)
            transformer_options[NEGPIP_MASK_KEY] = ns

        transformer_options[COND_UNCOND_COUPLE_KEY] = cond_or_uncond_couple

        return xs, cs, rope_emb, transformer_options

    def _cosmos_ppm_attn_output_ca_couple(transformer_options: dict, out: torch.Tensor):
        cond_or_uncond = transformer_options[COND_UNCOND_COUPLE_KEY]
        size = tuple(transformer_options["activations_shape"][-2:])
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

    return _cosmos_couple_diffusion_wrapper
