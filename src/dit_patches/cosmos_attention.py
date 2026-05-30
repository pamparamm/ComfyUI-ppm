from functools import partial
from typing import Any, Callable

import einops
import torch

from comfy.ldm.cosmos.predict2 import Attention as CosmosAttention
from comfy.ldm.cosmos.predict2 import MiniTrainDIT
from comfy.model_patcher import ModelPatcher

apply_rope_split_half: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
try:
    import comfy.quant_ops

    apply_rope_split_half = comfy.quant_ops.ck.apply_rope_split_half
except (ImportError, AttributeError):  # fmt: skip

    def _apply_rope_split_half(
        q: torch.Tensor,
        k: torch.Tensor,
        rope_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = _apply_rotary_pos_emb(q, rope_emb)
        k = _apply_rotary_pos_emb(k, rope_emb)
        return q, k

    # Copied from comfy.ldm.cosmos.predict2 for backward compatibility
    def _apply_rotary_pos_emb(
        t: torch.Tensor,
        freqs: torch.Tensor,
    ) -> torch.Tensor:
        t_ = t.reshape(*t.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2).float()
        t_out = freqs[..., 0] * t_[..., 0] + freqs[..., 1] * t_[..., 1]
        t_out = t_out.movedim(-1, -2).reshape(*t.shape).type_as(t)
        return t_out

    apply_rope_split_half = _apply_rope_split_half


def patch_cosmos_attention(model_patcher: ModelPatcher):
    cosmos_model: MiniTrainDIT = model_patcher.get_model_object("diffusion_model")  # type: ignore
    for block_name, block in (
        (n, b)
        for n, b in cosmos_model.named_modules()
        if ("cross_attn" in n or "self_attn" in n) and isinstance(b, CosmosAttention)
    ):
        patch_name = f"diffusion_model.{block_name}.forward"
        if patch_name not in model_patcher.object_patches:
            model_patcher.add_object_patch(patch_name, partial(_forward_patched, block))


def _forward_patched(
    self: CosmosAttention,
    x: torch.Tensor,
    context: torch.Tensor | None = None,
    rope_emb: torch.Tensor | None = None,
    transformer_options: dict | None = {},
) -> torch.Tensor:
    transformer_options = transformer_options if transformer_options is not None else {}

    x, context, rope_emb, transformer_options = _calc_patched(
        "ppm_attn_pre_sa" if context is None else "ppm_attn_pre_ca",
        transformer_options,
        None,
        x,
        context,
        rope_emb,
        transformer_options,
    )

    q, k, v = _compute_qkv_patched(
        self,
        x,
        context,
        rope_emb=rope_emb,
        transformer_options=transformer_options,  # type: ignore
    )
    output = self.compute_attention(q, k, v, transformer_options=transformer_options)

    output = _calc_patched(
        "ppm_attn_output_sa" if context is None else "ppm_attn_output_ca",
        transformer_options,  # type: ignore
        None,
        output,
    )

    return output


def _compute_qkv_patched(
    self: CosmosAttention,
    x: torch.Tensor,
    context: torch.Tensor | None = None,
    rope_emb: torch.Tensor | None = None,
    transformer_options: dict[str, Any] = {},
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if context is not None:
        q = _calc_patched("ppm_attn_q_proj_ca", transformer_options, self.q_proj, x)
        k = _calc_patched("ppm_attn_k_proj_ca", transformer_options, self.k_proj, context)
        v = _calc_patched("ppm_attn_v_proj_ca", transformer_options, self.v_proj, context)
    else:
        q = _calc_patched("ppm_attn_q_proj_sa", transformer_options, self.q_proj, x)
        k = _calc_patched("ppm_attn_k_proj_sa", transformer_options, self.k_proj, x)
        v = _calc_patched("ppm_attn_v_proj_sa", transformer_options, self.v_proj, x)

    q, k, v = map(
        lambda t: einops.rearrange(t, "b ... (h d) -> b ... h d", h=self.n_heads, d=self.head_dim),
        (q, k, v),
    )

    def apply_norm_and_rotary_pos_emb_patched(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, rope_emb: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = _calc_patched("ppm_attn_q_norm", transformer_options, self.q_norm, q)
        k = _calc_patched("ppm_attn_k_norm", transformer_options, self.k_norm, k)
        v = _calc_patched("ppm_attn_v_norm", transformer_options, self.v_norm, v)
        if self.is_selfattn and rope_emb is not None:  # only apply to self-attention!
            q, k = _calc_patched("ppm_attn_rope", transformer_options, apply_rope_split_half, q, k, rope_emb)
        return q, k, v

    q, k, v = apply_norm_and_rotary_pos_emb_patched(q, k, v, rope_emb)

    return q, k, v


def _calc_patched(patch_key: str, transformer_options: dict[str, Any], func: Callable | None, *args, **kwargs):
    patch: Callable = transformer_options.get(patch_key)  # type: ignore
    if patch is not None:
        if func is not None:
            return patch(func, transformer_options, *args, **kwargs)
        return patch(transformer_options, *args, **kwargs)
    if func is not None:
        return func(*args, **kwargs)
    if len(args) == 0:
        return
    if len(args) == 1:
        return args[0]
    return args
