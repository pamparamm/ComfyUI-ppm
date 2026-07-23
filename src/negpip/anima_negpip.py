from typing import Any, Callable

import torch

import comfy.conds
import comfy.patcher_extension
from comfy.model_patcher import ModelPatcher

WRAPPER_KEY = "ppm_negpip_anima"
COND_NEGPIP_MASK_KEY = "c_ppm_negpip_mask"
NEGPIP_MASK_KEY = "ppm_negpip_mask"


def patch_anima_negpip(patcher: ModelPatcher):
    previous_extra_conds = patcher.get_model_object("extra_conds")
    patcher.add_object_patch(
        "extra_conds",
        anima_extra_conds_negpip_wrapper(previous_extra_conds),
    )
    patcher.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        WRAPPER_KEY,
        cosmos_diffusion_negpip_wrapper,
    )
    patcher.set_model_attn2_patch(cosmos_attn2_negpip)


def anima_extra_conds_negpip_wrapper(
    previous_extra_conds: Callable[..., dict],
):
    def _anima_extra_conds_negpip_wrapper(**kwargs):
        t5xxl_weights: torch.Tensor = kwargs.get("t5xxl_weights", None)  # type: ignore
        negpip_mask = None
        if t5xxl_weights is not None:
            t5xxl_weights_abs = torch.abs(t5xxl_weights)
            negpip_positions = t5xxl_weights == t5xxl_weights_abs
            negpip_mask = negpip_positions.int()
            negpip_mask[negpip_mask == 0] = -1
            negpip_mask = negpip_mask.unsqueeze(0).unsqueeze(-1)

            if negpip_mask.shape[1] < 512:
                negpip_mask = torch.nn.functional.pad(negpip_mask, (0, 0, 0, 512 - negpip_mask.shape[1]), value=1.0)

            kwargs["t5xxl_weights"] = t5xxl_weights_abs

        out = previous_extra_conds(**kwargs)
        if negpip_mask is not None:
            out[COND_NEGPIP_MASK_KEY] = comfy.conds.CONDRegular(negpip_mask)

        return out

    return _anima_extra_conds_negpip_wrapper


def cosmos_diffusion_negpip_wrapper(executor, *args, **kwargs):
    context: torch.Tensor = args[2]
    transformer_options: dict[str, Any] = kwargs.get("transformer_options", {})
    negpip_mask: torch.Tensor | None = kwargs.get(COND_NEGPIP_MASK_KEY)

    if negpip_mask is not None:
        transformer_options[NEGPIP_MASK_KEY] = negpip_mask.to(context)

    kwargs["transformer_options"] = transformer_options

    return executor(*args, **kwargs)


def cosmos_attn2_negpip(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pe: torch.Tensor | None = None,
    attn_mask: torch.Tensor | None = None,
    extra_options: dict = {},
) -> dict[str, torch.Tensor | None]:
    negpip_mask = extra_options.get(NEGPIP_MASK_KEY) if extra_options else None
    new_v = v if negpip_mask is None else v * negpip_mask
    return {"q": q, "k": k, "v": new_v, "pe": pe}
