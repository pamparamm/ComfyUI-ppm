from functools import partial
from math import copysign
from types import ModuleType

import numpy as np
import torch

from .module_injector import get_module_injector
from .utils import from_zero

INITIALIZED = False


def _advanced_encode_from_tokens_negpip_wrapper(advanced_encode_from_tokens, from_zero):
    def advanced_encode_from_tokens_negpip(
        tokenized,
        token_normalization,
        weight_interpretation,
        encode_func,
        m_token="+",
        w_max=1.0,
        return_pooled=False,
        apply_to_pooled=False,
        tokenizer=None,
        **extra_args,
    ):
        tokenized_abs = [[(t, abs(w), p) for t, w, p in x] for x in tokenized]
        weights_sign = [[copysign(1, w) for _, w, _ in x] for x in tokenized]

        def _encoded_with_negpip():
            token = tokenizer.tokenize_with_weights(m_token)[0][tokenizer.tokens_start]
            max_length = tokenizer.max_length

            tokens = [[token for _ in range(max_length)]]
            emb, _ = encode_func(tokens)
            if emb.shape[1] == max_length:
                return False
            if emb.shape[1] == max_length * 2:
                return True
            raise ValueError(
                f"Unknown embedding shape: expected {max_length} or {max_length * 2}, but found {emb.shape[1]}. Perhaps you've applied NegPiP node more than once?"
            )

        encoded_with_negpip = _encoded_with_negpip()

        def _encode_func(tokens):
            emb, pooled = encode_func(tokens)
            if encoded_with_negpip:
                return emb[:, 0::2, :], pooled
            return emb, pooled

        def _apply_negpip(weights_sign, emb):
            emb_negpip = torch.empty_like(emb).repeat(1, 2, 1)
            emb_negpip[:, 0::2, :] = emb
            emb_negpip[:, 1::2, :] = from_zero(weights_sign, emb)
            return emb_negpip

        weighted_emb, pooled = advanced_encode_from_tokens(
            tokenized_abs,
            token_normalization,
            weight_interpretation,
            _encode_func,
            m_token,
            w_max,
            return_pooled,
            apply_to_pooled,
            tokenizer,
            **extra_args,
        )

        if encoded_with_negpip:
            weighted_emb = _apply_negpip(weights_sign, weighted_emb)

        return weighted_emb, pooled

    return advanced_encode_from_tokens_negpip


def _encode_regions_negpip_wrapper(create_masked_prompt):
    def encode_regions_negpip(clip_regions, encode, tokenizer):
        base_weighted_tokens = clip_regions["base_tokens"]
        start_from_masked = clip_regions["start_from_masked"]
        mask_token = clip_regions["mask_token"]
        strict_mask = clip_regions["strict_mask"]

        # calc base embedding
        base_embedding_full, pool = encode(base_weighted_tokens)

        # Avoid numpy value error and passthrough base embeddings if no regions are set.

        # calc global target mask
        global_target_mask = np.any(np.stack(clip_regions["targets"]), axis=0).astype(int)

        # calc global region mask
        global_region_mask = np.any(np.stack(clip_regions["regions"]), axis=0).astype(float)
        regions_sum = np.sum(np.stack(clip_regions["regions"]), axis=0)
        regions_normalized = np.divide(1, regions_sum, out=np.zeros_like(regions_sum), where=regions_sum != 0)

        # mask base embeddings
        base_masked_prompt = create_masked_prompt(base_weighted_tokens, global_target_mask, mask_token)
        base_embedding_masked, _ = encode(base_masked_prompt)
        base_embedding_start = base_embedding_full * (1 - start_from_masked) + base_embedding_masked * start_from_masked
        base_embedding_outer = base_embedding_full * (1 - strict_mask) + base_embedding_masked * strict_mask

        region_embeddings = []
        for region, target, weight in zip(clip_regions["regions"], clip_regions["targets"], clip_regions["weights"]):
            region_masking = torch.tensor(
                regions_normalized * region * weight, dtype=base_embedding_full.dtype, device=base_embedding_full.device
            ).unsqueeze(-1)

            region_prompt = create_masked_prompt(base_weighted_tokens, global_target_mask - target, mask_token)
            region_emb, _ = encode(region_prompt)
            region_emb -= base_embedding_start
            if region_emb.shape[1] == 2 * region_masking.shape[1]:
                region_masking = torch.repeat_interleave(region_masking, 2, dim=1)
            region_emb *= region_masking

            region_embeddings.append(region_emb)
        region_embeddings = torch.stack(region_embeddings).sum(axis=0)

        embeddings_final_mask = torch.tensor(
            global_region_mask, dtype=base_embedding_full.dtype, device=base_embedding_full.device
        ).unsqueeze(-1)
        if region_embeddings.shape[1] == 2 * embeddings_final_mask.shape[1]:
            embeddings_final_mask = torch.repeat_interleave(embeddings_final_mask, 2, dim=1)

        embeddings_final = base_embedding_start * embeddings_final_mask + base_embedding_outer * (
            1 - embeddings_final_mask
        )
        embeddings_final += region_embeddings
        return embeddings_final, pool

    return encode_regions_negpip


def patch_prompt_control():
    global INITIALIZED

    if not INITIALIZED:
        try:
            injector = get_module_injector("comfyui-prompt-control")

            def _patch(module: ModuleType):
                adv_encode = module.prompt_control.adv_encode
                prompts = module.prompt_control.prompts
                cutoff = module.prompt_control.cutoff
                advanced_encode_from_tokens_negpip = _advanced_encode_from_tokens_negpip_wrapper(
                    adv_encode.advanced_encode_from_tokens, from_zero
                )
                encode_regions_negpip = _encode_regions_negpip_wrapper(cutoff.create_masked_prompt)

                def _make_patch_negpip(te_name, orig_fn, normalization, style, extra):
                    def encode(t):
                        r = advanced_encode_from_tokens_negpip(
                            t, normalization, style, orig_fn, return_pooled=True, apply_to_pooled=False, **extra
                        )
                        return prompts.apply_weights(r, te_name, extra.get("clip_weights"))

                    if "cuts" in extra:
                        return partial(prompts.process_cuts, encode, extra)
                    return encode

                prompts.make_patch = _make_patch_negpip
                cutoff.encode_regions = encode_regions_negpip

            injector.patch(_patch)

        finally:
            INITIALIZED = True
