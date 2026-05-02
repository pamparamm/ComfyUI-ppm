from typing import Any, Callable

import einops
import torch

import comfy.conds
from comfy.ldm.cosmos.predict2 import Attention as CosmosAttention
from comfy.ldm.cosmos.predict2 import apply_rotary_pos_emb


COND_NEGPIP_MASK_KEY = "c_negpip_mask"
NEGPIP_MASK_KEY = "negpip_mask"


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
        transformer_options[NEGPIP_MASK_KEY] = negpip_mask.to(context)

    kwargs["transformer_options"] = transformer_options

    return executor(*args, **kwargs)


def cosmos_attention_forward_negpip(
    self: CosmosAttention,
    x: torch.Tensor,
    context: torch.Tensor | None = None,
    rope_emb: torch.Tensor | None = None,
    transformer_options: dict | None = {},
) -> torch.Tensor:
    negpip_mask = transformer_options.get(NEGPIP_MASK_KEY) if transformer_options else None
    q, k, v = cosmos_attention_compute_qkv_negpip(
        self,
        x,
        context,
        rope_emb=rope_emb,
        negpip_mask=negpip_mask,
    )
    return self.compute_attention(q, k, v, transformer_options=transformer_options)


def cosmos_attention_compute_qkv_negpip(
    self: CosmosAttention,
    x: torch.Tensor,
    context: torch.Tensor | None = None,
    rope_emb: torch.Tensor | None = None,
    negpip_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = self.q_proj(x)
    context_k = x if context is None else context
    context_v = context_k if negpip_mask is None else context_k * negpip_mask
    k = self.k_proj(context_k)
    v = self.v_proj(context_v)

    q, k, v = map(
        lambda t: einops.rearrange(t, "b ... (h d) -> b ... h d", h=self.n_heads, d=self.head_dim),
        (q, k, v),
    )

    def apply_norm_and_rotary_pos_emb(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, rope_emb: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)
        if self.is_selfattn and rope_emb is not None:  # only apply to self-attention!
            q = apply_rotary_pos_emb(q, rope_emb)
            k = apply_rotary_pos_emb(k, rope_emb)
        return q, k, v

    q, k, v = apply_norm_and_rotary_pos_emb(q, k, v, rope_emb)

    return q, k, v
