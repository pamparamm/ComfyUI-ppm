import copy
import logging
import math
from typing import Callable

import torch

import comfy.conds
import comfy.patcher_extension
import comfy.text_encoders.qwen_vl
from comfy.ldm.moss_image.model import MOSSImage as MOSSImageDIT
from comfy.model_patcher import ModelPatcher
from comfy.sd import CLIP

WRAPPER_KEY = "ppm_negpip_moss"
TOKEN_WEIGHTS_KEY = "ppm_negpip_text_weights"
COND_NEGPIP_BIAS_KEY = "c_ppm_negpip_attn_bias"
COND_NEGPIP_V_SCALE_KEY = "c_ppm_negpip_v_scale"
NEGPIP_WEIGHTING_KEY = "ppm_negpip_weighting"


def entry_rows(token) -> int:
    if not isinstance(token, dict):
        return 1
    _, grid = comfy.text_encoders.qwen_vl.process_qwen2vl_images(token["data"], patch_size=16)  # 2x2 patches merge
    return int(grid[0, 1] * grid[0, 2]) // 4


def patch_moss_negpip(patcher: ModelPatcher, clip: CLIP, encoder: str):
    clip.tokenizer = moss_tokenizer_negpip(clip.tokenizer, encoder)
    clip.patcher.add_object_patch(
        f"{encoder}.encode_token_weights",
        moss_encode_token_weights_negpip_wrapper(clip.patcher.get_model_object(f"{encoder}.encode_token_weights")),
    )

    previous_extra_conds = patcher.get_model_object("extra_conds")
    patcher.add_object_patch(
        "extra_conds",
        moss_extra_conds_negpip_wrapper(previous_extra_conds),
    )
    patcher.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        WRAPPER_KEY,
        moss_diffusion_negpip_wrapper(patcher),
    )


def moss_tokenizer_negpip(tokenizer, encoder: str):
    tok = copy.copy(tokenizer)
    inner = copy.copy(getattr(tok, encoder))
    tokenize = inner.tokenize_with_weights

    def tokenize_with_weights(text: str, return_word_ids: bool = False, **kwargs):
        kwargs["disable_weights"] = False
        return tokenize(text, return_word_ids=return_word_ids, **kwargs)

    inner.tokenize_with_weights = tokenize_with_weights
    setattr(tok, encoder, inner)
    return tok


def moss_encode_token_weights_negpip_wrapper(previous_encode_token_weights: Callable):
    def _moss_encode_token_weights_negpip_wrapper(token_weight_pairs):
        weights = [w for section in token_weight_pairs for token, w, *_ in section for _ in range(entry_rows(token))]
        unweighted = [[(t, 1.0, *rest) for t, _, *rest in section] for section in token_weight_pairs]

        out = previous_encode_token_weights(unweighted)

        extra = out[2] if len(out) > 2 else {}
        extra[TOKEN_WEIGHTS_KEY] = torch.tensor(weights, dtype=torch.float32).reshape(1, -1)

        return (*out[:2], extra)

    return _moss_encode_token_weights_negpip_wrapper


def moss_extra_conds_negpip_wrapper(previous_extra_conds: Callable[..., dict]):
    def _moss_extra_conds_negpip_wrapper(**kwargs):
        weights: torch.Tensor | None = kwargs.pop(TOKEN_WEIGHTS_KEY, None)
        cross_attn = kwargs.get("cross_attn", None)
        n_text = 0 if cross_attn is None else cross_attn.shape[1]

        if weights is not None and weights.shape[-1] < n_text:
            pad = weights.new_ones(*weights.shape[:-1], n_text - weights.shape[-1])
            weights = torch.cat((weights, pad), dim=-1)
        elif weights is not None and weights.shape[-1] > n_text:
            logging.warning(
                "NegPiP: {} prompt weights do not fit {} text tokens, ignoring weights.".format(
                    weights.shape[-1], n_text
                )
            )
            weights = None

        out = previous_extra_conds(**kwargs)

        if weights is not None and not torch.all(weights == 1.0):
            neg = weights < 0
            bias = weights.abs().clamp_min(1e-4).log()
            bias = bias - (bias.logsumexp(-1, keepdim=True) - math.log(bias.shape[-1]))
            if torch.any(bias):
                out[COND_NEGPIP_BIAS_KEY] = comfy.conds.CONDRegular(bias.unsqueeze(1).unsqueeze(1))
            if torch.any(neg):
                v_scale = torch.ones_like(weights)
                v_scale[neg] = -1.0
                out[COND_NEGPIP_V_SCALE_KEY] = comfy.conds.CONDRegular(v_scale.unsqueeze(-1))

        return out

    return _moss_extra_conds_negpip_wrapper


def moss_diffusion_negpip_wrapper(patcher: ModelPatcher):
    def _moss_diffusion_negpip_wrapper(executor, latent, timestep, context, transformer_options, **kwargs):
        bias: torch.Tensor | None = kwargs.get(COND_NEGPIP_BIAS_KEY)
        v_scale: torch.Tensor | None = kwargs.get(COND_NEGPIP_V_SCALE_KEY)
        if bias is None and v_scale is None:
            return executor(latent, timestep, context, transformer_options, **kwargs)

        n_img = latent.shape[2] * latent.shape[3]
        if bias is not None:
            bias = torch.cat((bias.new_zeros(*bias.shape[:3], n_img), bias), dim=-1)

        diffusion_model: MOSSImageDIT = patcher.get_model_object("diffusion_model")  # type: ignore
        attentions = [block.comfy_attention.function for block in diffusion_model.layers]  # pyrefly: ignore[missing-attribute]
        transformer_options[NEGPIP_WEIGHTING_KEY] = {
            "n_img": n_img,
            "n_text": context.shape[1],
            "bias": bias,
            "v_scale": v_scale,
            "attentions": attentions,
            "previous": transformer_options.get("optimized_attention_override"),
        }
        transformer_options["optimized_attention_override"] = moss_attention_negpip

        return executor(latent, timestep, context, transformer_options, **kwargs)

    return _moss_diffusion_negpip_wrapper


def moss_attention_negpip(
    default_attention: Callable,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    mask: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    transformer_options = kwargs["transformer_options"]
    weighting = transformer_options[NEGPIP_WEIGHTING_KEY]
    n_img, n_text = weighting["n_img"], weighting["n_text"]
    attentions = weighting["attentions"]
    block_index = transformer_options.get("block_index", 0)
    attention = attentions[block_index] if 0 <= block_index < len(attentions) else attentions[0]
    attention = attention or default_attention
    previous = weighting["previous"]

    def call(q_part: torch.Tensor, v_part: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        if bias is None:
            bias = mask
        elif mask is not None and bias.shape[-1] == mask.shape[-1]:
            bias = bias + mask
        if previous is not None:
            return previous(default_attention, q_part, k, v_part, heads, mask=bias, **kwargs)
        return attention(q_part, k, v_part, heads, mask=bias, **kwargs)

    n_ref = k.shape[1] - n_img - n_text
    if n_ref < 0:
        return call(q, v, None)

    bias = weighting["bias"]
    if bias is not None and n_ref > 0:
        bias = torch.cat((bias, bias.new_zeros(*bias.shape[:3], n_ref)), dim=-1)

    v_scale = weighting["v_scale"]
    v_weighted = v
    if v_scale is not None:
        v_weighted = torch.cat((v[:, :n_img], v[:, n_img : n_img + n_text] * v_scale, v[:, n_img + n_text :]), dim=1)

    return torch.cat((call(q[:, :n_img], v_weighted, bias), call(q[:, n_img:], v, None)), dim=1)
