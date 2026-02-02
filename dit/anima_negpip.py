from typing import Any, Callable

import einops
import torch

import comfy.conds
from comfy.ldm.cosmos.predict2 import Attention as CosmosAttention
from comfy.ldm.cosmos.predict2 import apply_rotary_pos_emb
from comfy.model_base import Anima


def anima_extra_conds_negpip(
    self: Anima,
    **kwargs,
):
    out = super(Anima, self).extra_conds(**kwargs)
    cross_attn = kwargs.get("cross_attn", None)
    t5xxl_ids = kwargs.get("t5xxl_ids", None)
    t5xxl_weights = kwargs.get("t5xxl_weights", None)
    device = kwargs["device"]
    if cross_attn is not None:
        if t5xxl_ids is not None:
            cross_attn = self.diffusion_model.preprocess_text_embeds(  # pyright: ignore[reportCallIssue]
                cross_attn.to(device=device, dtype=self.get_dtype()), t5xxl_ids.unsqueeze(0).to(device=device)
            )
            negpip_mask = torch.ones_like(cross_attn)

            if t5xxl_weights is not None:
                t5xxl_weights: torch.Tensor = t5xxl_weights.unsqueeze(0).unsqueeze(-1).to(cross_attn)
                t5xxl_weights_abs = torch.abs(t5xxl_weights)

                negpip_mask = (t5xxl_weights == t5xxl_weights_abs).int()
                negpip_mask[negpip_mask == 0] = -1

                cross_attn *= t5xxl_weights_abs

            if cross_attn.shape[1] < 512:
                cross_attn = torch.nn.functional.pad(cross_attn, (0, 0, 0, 512 - cross_attn.shape[1]))
                negpip_mask = torch.nn.functional.pad(negpip_mask, (0, 0, 0, 512 - negpip_mask.shape[1]), value=1)

            out["c_negpip_mask"] = comfy.conds.CONDRegular(negpip_mask)

        out["c_crossattn"] = comfy.conds.CONDRegular(cross_attn)
    return out


def anima_forward_negpip(
    _forward: Callable[..., torch.Tensor],
    x: torch.Tensor,
    timesteps: torch.Tensor,
    context: torch.Tensor,
    fps: torch.Tensor | None = None,
    padding_mask: torch.Tensor | None = None,
    **kwargs,
):
    transformer_options: dict[str, Any] = kwargs.get("transformer_options", {})
    transformer_options["negpip_mask"] = kwargs.get("c_negpip_mask", torch.ones(context.shape[0], context.shape[1], 1))
    kwargs["transformer_options"] = transformer_options

    return _forward(x, timesteps, context, fps, padding_mask, **kwargs)


def cosmos_attention_forward_negpip(
    self: CosmosAttention,
    x: torch.Tensor,
    context: torch.Tensor | None = None,
    rope_emb: torch.Tensor | None = None,
    transformer_options: dict | None = {},
) -> torch.Tensor:
    negpip_mask = transformer_options.get("negpip_mask") if transformer_options else None
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
    context_v = context_k if negpip_mask is None else context_k * negpip_mask.to(context_k)
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
