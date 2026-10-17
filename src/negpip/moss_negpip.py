import copy
from typing import Callable

import torch

import comfy.conds
import comfy.patcher_extension
from comfy.ldm.moss_image.model import MOSSImage as MOSSImageDIT
from comfy.model_patcher import ModelPatcher
from comfy.sd import CLIP

WRAPPER_KEY = "ppm_negpip_moss"
TOKEN_WEIGHTS_KEY = "ppm_negpip_text_weights"
COND_NEGPIP_BIAS_KEY = "c_ppm_negpip_attn_bias"
COND_NEGPIP_V_SCALE_KEY = "c_ppm_negpip_v_scale"
NEGPIP_WEIGHTING_KEY = "ppm_negpip_weighting"

WEIGHT_GAIN = 3.0


def patch_moss_negpip(patcher: ModelPatcher, clip: CLIP, encoder: str):
    clip.tokenizer = moss_tokenizer_negpip(clip.tokenizer, encoder)
    clip.patcher.add_object_patch(
        f"{encoder}.encode_token_weights",
        moss_encode_token_weights_negpip_wrapper(getattr(clip.patcher.model, encoder).encode_token_weights),
    )

    previous_extra_conds = patcher.get_model_object("extra_conds")
    patcher.add_object_patch(
        "extra_conds",
        moss_extra_conds_negpip_wrapper(previous_extra_conds),
    )
    diffusion_model: MOSSImageDIT = patcher.get_model_object("diffusion_model")  # type: ignore
    attention = diffusion_model.layers[0].comfy_attention.function
    patcher.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        WRAPPER_KEY,
        moss_diffusion_negpip_wrapper(attention),
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
        weights = [w for section in token_weight_pairs for _, w, *_ in section]
        unweighted = [[(t, 1.0, *rest) for t, _, *rest in section] for section in token_weight_pairs]

        out = previous_encode_token_weights(unweighted)

        extra = out[2] if len(out) > 2 else {}
        extra[TOKEN_WEIGHTS_KEY] = torch.tensor(weights, dtype=torch.float32).reshape(1, -1)

        return (*out[:2], extra)

    return _moss_encode_token_weights_negpip_wrapper


def moss_extra_conds_negpip_wrapper(previous_extra_conds: Callable[..., dict]):
    def _moss_extra_conds_negpip_wrapper(**kwargs):
        weights: torch.Tensor | None = kwargs.pop(TOKEN_WEIGHTS_KEY, None)
        out = previous_extra_conds(**kwargs)

        if weights is not None and not torch.all(weights == 1.0):
            neg = weights < 0
            bias = weights.clamp_min(1e-4).log() * WEIGHT_GAIN
            bias[neg] = 0.0
            if torch.any(bias):
                out[COND_NEGPIP_BIAS_KEY] = comfy.conds.CONDRegular(bias.unsqueeze(1).unsqueeze(1))
            if torch.any(neg):
                v_scale = torch.where(neg, weights, torch.ones_like(weights))
                out[COND_NEGPIP_V_SCALE_KEY] = comfy.conds.CONDRegular(v_scale.unsqueeze(-1))

        return out

    return _moss_extra_conds_negpip_wrapper


def moss_diffusion_negpip_wrapper(attention: Callable | None):
    def _moss_diffusion_negpip_wrapper(executor, latent, timestep, context, transformer_options, **kwargs):
        bias: torch.Tensor | None = kwargs.get(COND_NEGPIP_BIAS_KEY)
        v_scale: torch.Tensor | None = kwargs.get(COND_NEGPIP_V_SCALE_KEY)
        if bias is None and v_scale is None:
            return executor(latent, timestep, context, transformer_options, **kwargs)

        n_img = latent.shape[2] * latent.shape[3]
        if bias is not None:
            bias = torch.cat((bias.new_zeros(*bias.shape[:3], n_img), bias), dim=-1)

        transformer_options[NEGPIP_WEIGHTING_KEY] = {
            "n_img": n_img,
            "n_text": context.shape[1],
            "bias": bias,
            "v_scale": v_scale,
            "attention": attention,
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
    **kwargs,
) -> torch.Tensor:
    weighting = kwargs["transformer_options"][NEGPIP_WEIGHTING_KEY]
    n_img, n_text = weighting["n_img"], weighting["n_text"]
    attention = weighting["attention"] or default_attention
    previous = weighting["previous"]

    def call(q_part: torch.Tensor, v_part: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if previous is not None:
            return previous(default_attention, q_part, k, v_part, heads, mask=mask, **kwargs)
        return attention(q_part, k, v_part, heads, mask=mask, **kwargs)

    if k.shape[1] != n_img + n_text:
        return call(q, v, None)

    v_scale = weighting["v_scale"]
    v_weighted = v if v_scale is None else torch.cat((v[:, :-n_text], v[:, -n_text:] * v_scale), dim=1)

    return torch.cat((call(q[:, :n_img], v_weighted, weighting["bias"]), call(q[:, n_img:], v, None)), dim=1)
