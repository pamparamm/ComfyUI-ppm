from typing import Any, Callable

import torch

import comfy.conds

COND_NEGPIP_MASK_KEY = "c_ppm_negpip_mask"
NEGPIP_MASK_KEY = "ppm_negpip_mask"


def anima_extra_conds_negpip(
    extra_conds: Callable[..., dict],
    **kwargs,
):
    t5xxl_weights = kwargs.get("t5xxl_weights", None)
    negpip_mask = None
    if t5xxl_weights is not None:
        t5xxl_weights_abs = torch.abs(t5xxl_weights)

        negpip_mask = (t5xxl_weights == t5xxl_weights_abs).int()
        negpip_mask[negpip_mask == 0] = -1
        negpip_mask = negpip_mask.unsqueeze(0).unsqueeze(-1)

        if negpip_mask.shape[1] < 512:
            negpip_mask = torch.nn.functional.pad(negpip_mask, (0, 0, 0, 512 - negpip_mask.shape[1]), value=1.0)

        kwargs["t5xxl_weights"] = t5xxl_weights_abs

    out = extra_conds(**kwargs)
    if negpip_mask is not None:
        out[COND_NEGPIP_MASK_KEY] = comfy.conds.CONDRegular(negpip_mask)

    return out


def cosmos_diffusion_negpip_wrapper(executor, *args, **kwargs):
    context: torch.Tensor = args[2]
    transformer_options: dict[str, Any] = kwargs.get("transformer_options", {})
    negpip_mask: torch.Tensor | None = kwargs.get(COND_NEGPIP_MASK_KEY)

    if negpip_mask is not None:
        transformer_options["ppm_attn_v_proj_ca"] = _cosmos_ppm_attn_v_proj_ca_negpip
        transformer_options[NEGPIP_MASK_KEY] = negpip_mask.to(context)

    kwargs["transformer_options"] = transformer_options

    return executor(*args, **kwargs)


def _cosmos_ppm_attn_v_proj_ca_negpip(
    func: Callable, transformer_options: dict, context: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    negpip_mask = transformer_options.get(NEGPIP_MASK_KEY) if transformer_options else None
    context_v = context if negpip_mask is None else context * negpip_mask
    return func(context_v)
