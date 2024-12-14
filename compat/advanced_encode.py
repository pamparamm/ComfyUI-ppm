import importlib
import torch
from functools import partial
from math import copysign

INITIALIZED = False


def from_zero(weights, base_emb):
    weight_tensor = torch.tensor(weights, dtype=base_emb.dtype, device=base_emb.device)
    weight_tensor = weight_tensor.reshape(1, -1, 1).expand(base_emb.shape)
    return base_emb * weight_tensor


def _advanced_encode_from_tokens_negpip_wrapper(advanced_encode_from_tokens, from_zero):

    def advanced_encode_from_tokens_negpip(
        tokenized,
        token_normalization,
        weight_interpretation,
        encode_func,
        m_token=266,
        length=77,
        w_max=1.0,
        return_pooled=False,
        apply_to_pooled=False,
        **extra_args,
    ):
        tokenized_abs = [[(t, abs(w), p) for t, w, p in x] for x in tokenized]
        weights_sign = [[copysign(1, w) for _, w, _ in x] for x in tokenized]

        def _encoded_with_negpip(encode_func, m_token=266, length=77):
            tokens = [[(m_token, 1.0) for _ in range(length)]]
            emb, _ = encode_func(tokens)
            if emb.shape[1] == length:
                return False
            elif emb.shape[1] == length * 2:
                return True
            raise ValueError("Unknown tensor shape - perhaps you've applied NegPip node more than once")

        encoded_with_negpip = _encoded_with_negpip(encode_func, m_token, length)

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
            length,
            w_max,
            return_pooled,
            apply_to_pooled,
            **extra_args,
        )

        if encoded_with_negpip:
            weighted_emb = _apply_negpip(weights_sign, weighted_emb)

        return weighted_emb, pooled

    return advanced_encode_from_tokens_negpip


def hijack_adv_encode():
    global INITIALIZED

    def _hijack_ADV_CLIP_emb():
        try:
            import custom_nodes.ComfyUI_ADV_CLIP_emb.adv_encode as adv_encode
            import ComfyUI_ADV_CLIP_emb.adv_encode as adv_encode_inner

            advanced_encode_from_tokens_negpip = _advanced_encode_from_tokens_negpip_wrapper(
                adv_encode.advanced_encode_from_tokens, adv_encode.from_zero
            )

            adv_encode.advanced_encode_from_tokens = advanced_encode_from_tokens_negpip
            adv_encode_inner.advanced_encode_from_tokens = advanced_encode_from_tokens_negpip

        except ImportError:
            pass

    def _hijack_prompt_control():
        try:
            adv_encode = importlib.import_module("custom_nodes.comfyui-prompt-control.prompt_control.adv_encode")
            prompts = importlib.import_module("custom_nodes.comfyui-prompt-control.prompt_control.prompts")
            prompts_inner = importlib.import_module("comfyui-prompt-control.prompt_control.prompts")

            advanced_encode_from_tokens_negpip = _advanced_encode_from_tokens_negpip_wrapper(adv_encode.advanced_encode_from_tokens, from_zero)

            def _make_patch_negpip(te_name, orig_fn, normalization, style, extra):
                def encode(t):
                    r = advanced_encode_from_tokens_negpip(t, normalization, style, orig_fn, return_pooled=True, apply_to_pooled=False, **extra)
                    return prompts.apply_weights(r, te_name, extra.get("clip_weights"))

                if "cuts" in extra:
                    return partial(prompts.process_cuts, encode, extra)
                return encode

            prompts.make_patch = _make_patch_negpip
            prompts_inner.make_patch = _make_patch_negpip

        except ImportError:
            pass

    if not INITIALIZED:
        import sys
        import pathlib

        custom_nodes = pathlib.Path(__file__).parent.parent.parent
        assert custom_nodes.name == "custom_nodes"

        sys.path.insert(0, str(custom_nodes))

        try:
            _hijack_ADV_CLIP_emb()
            _hijack_prompt_control()

        finally:
            sys.path.pop(0)
            INITIALIZED = True
