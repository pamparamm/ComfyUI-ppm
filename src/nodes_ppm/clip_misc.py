import json
import re

from comfy.comfy_types.node_typing import IO, ComfyNodeABC, InputTypeDict
from comfy.sd import CLIP
from comfy.sd1_clip import SDTokenizer
from node_helpers import conditioning_set_values
from nodes import (
    MAX_RESOLUTION,
    ConditioningCombine,
    ConditioningConcat,
    ConditioningSetTimestepRange,
    ConditioningZeroOut,
)


def get_special_tokens_map(clip: CLIP) -> dict[str, set[int]]:
    tokenizers: list[SDTokenizer] = [t for t in clip.tokenizer.__dict__.values() if isinstance(t, SDTokenizer)]
    special_tokens_map: dict[str, set[int]] = dict(
        (
            tokenizer.embedding_key.replace("clip_", ""),
            set([
                token
                for token in [tokenizer.start_token, tokenizer.end_token, tokenizer.pad_token]
                if isinstance(token, int)
            ]),
        )
        for tokenizer in tokenizers
    )
    return special_tokens_map


def get_token_dictionaries(clip: CLIP) -> dict[str, dict[int, str]]:
    tokenizers: list[SDTokenizer] = [t for t in clip.tokenizer.__dict__.values() if isinstance(t, SDTokenizer)]
    token_dictionaries: dict[str, dict[int, str]] = {}
    for tokenizer in tokenizers:
        t_key = tokenizer.embedding_key.replace("clip_", "")
        inv_vocab: dict[int, str] = tokenizer.inv_vocab
        token_dictionaries[t_key] = inv_vocab

    return token_dictionaries


class CLIPTextEncodeBREAK(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "clip": (IO.CLIP, {}),
                "text": (IO.STRING, {"multiline": True, "dynamicPrompts": True}),
            }
        }

    RETURN_TYPES = (IO.CONDITIONING,)
    FUNCTION = "encode"

    CATEGORY = "conditioning"

    SEPARATOR = "BREAK"

    def encode(self, clip: CLIP, text: str):
        concat_node = ConditioningConcat()

        chunks = text.split(self.SEPARATOR)

        cond_concat = None
        for chunk in chunks:
            chunk = chunk.strip()
            tokens = clip.tokenize(chunk)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            conditioning = [[cond, {"pooled_output": pooled}]]
            cond_concat = concat_node.concat(cond_concat, conditioning)[0] if cond_concat else conditioning

        return (cond_concat,)


class CLIPMicroConditioning(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "cond": (IO.CONDITIONING, {}),
                "width": (IO.INT, {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
                "height": (IO.INT, {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
                "crop_w": (IO.INT, {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
                "crop_h": (IO.INT, {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
                "target_width": (IO.INT, {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
                "target_height": (IO.INT, {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            }
        }

    RETURN_TYPES = (IO.CONDITIONING,)
    FUNCTION = "micro_conditioning"

    CATEGORY = "advanced/conditioning"

    def micro_conditioning(
        self, cond, width: int, height: int, crop_w: int, crop_h: int, target_width: int, target_height: int
    ):
        c = conditioning_set_values(
            cond,
            {
                "width": width,
                "height": height,
                "crop_w": crop_w,
                "crop_h": crop_h,
                "target_width": target_width,
                "target_height": target_height,
            },
        )
        return (c,)


class CLIPTokenCounter(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "clip": (IO.CLIP, {}),
                "text": (IO.STRING, {"multiline": True}),
            }
        }

    RETURN_TYPES = (
        IO.STRING,
        IO.STRING,
        IO.STRING,
    )
    RETURN_NAMES = (
        "count",
        "count_advanced",
        "parsed_tokens",
    )
    FUNCTION = "count"

    OUTPUT_NODE = True

    def count(self, clip: CLIP, text: str):
        special_tokens_map = get_special_tokens_map(clip)
        token_dictionaries = get_token_dictionaries(clip)
        tokens_map: dict[str, list[list[tuple[int, str]]]] = dict((key, []) for key in special_tokens_map.keys())

        prompts = self._parse_prompts(text)
        for prompt in prompts:
            if len(prompt) > 0:
                tokenizer_results: dict[str, list] = clip.tokenize(prompt)
                for key, tokens in [
                    (key, tokens[0]) for key, tokens in tokenizer_results.items() if key in special_tokens_map
                ]:
                    prompt_tokens = [
                        (t[0], token_dictionaries[key][t[0]]) for t in tokens if t[0] not in special_tokens_map[key]
                    ]
                    tokens_map[key].append(prompt_tokens)

        count_map = self._get_count(tokens_map)
        return (self._format_count(count_map), self._dump(count_map), self._format_tokens(tokens_map))

    @classmethod
    # TODO Rewrite into more robust function
    def _parse_prompts(cls, text: str) -> list[str]:
        text = re.sub(r"STYLE\(.*?\)", "", text)  # sanitize prompt_control STYLE(*)
        text = re.sub(r"\[(.*?)(\:.*?)+\]", r"\g<1>", text)  # replace prompt_control brackets
        prompts = text.split("BREAK")  # split text by BREAK keyword
        return prompts

    @classmethod
    def _get_count(cls, tokens_map: dict[str, list[list[tuple[int, str]]]]) -> dict[str, list[int]]:
        count: dict[str, list[int]] = {}
        for key, prompt_tokens in tokens_map.items():
            count[key] = [len(t) for t in prompt_tokens]
        return count

    @classmethod
    def _format_count(cls, count_map: dict[str, list[int]]) -> str:
        if len(count_map) == 0:
            return "0"
        count_key_to_clip: dict[str, list[str]] = {}
        for key, count in count_map.items():
            count_key = str(count)
            if count_key not in count_key_to_clip:
                count_key_to_clip[count_key] = []
            count_key_to_clip[count_key].append(key)
        if len(count_key_to_clip) == 1:
            count_key = list(count_key_to_clip.keys())[0]
            count_simple = count_key.removeprefix("[").removesuffix("]")
            if len(count_simple) > 0:
                return count_simple
            return "0"
        return cls._dump(count_map)  # fallback to json dump

    @classmethod
    def _format_tokens(cls, tokens_map: dict[str, list[list[tuple[int, str]]]]) -> str:
        formatted_map = dict(((k, [[f"`{t[1]}` ({t[0]})`" for t in p] for p in v]) for k, v in tokens_map.items()))
        return cls._dump(formatted_map)

    @classmethod
    def _dump(cls, count_map: dict) -> str:
        return json.dumps(count_map, indent=2)


class ConditioningZeroOutCombine(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "conditioning": (IO.CONDITIONING, {}),
                "zero_out_end": (IO.FLOAT, {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = (IO.CONDITIONING,)
    FUNCTION = "zero_out_combine"

    CATEGORY = "advanced/conditioning"

    def zero_out_combine(self, conditioning, zero_out_end: float):
        zero_out_node = ConditioningZeroOut()
        timestep_node = ConditioningSetTimestepRange()
        combine_node = ConditioningCombine()

        c_zero = zero_out_node.zero_out(conditioning)[0]
        c_zero = timestep_node.set_range(c_zero, 0.0, zero_out_end)[0]
        c = timestep_node.set_range(conditioning, zero_out_end, 1.0)[0]
        c_combined = combine_node.combine(c, c_zero)
        return c_combined


class CLIPTextEncodeInvertWeights(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "clip": (IO.CLIP, {}),
                "text": (IO.STRING, {"multiline": True, "dynamicPrompts": True}),
                "invert_special_tokens": (IO.BOOLEAN, {"default": False}),
            }
        }

    RETURN_TYPES = (IO.CONDITIONING,)
    FUNCTION = "encode"

    CATEGORY = "conditioning"

    SEPARATOR = "BREAK"

    def encode(self, clip: CLIP, text: str, invert_special_tokens: bool):
        concat_node = ConditioningConcat()
        special_tokens_map = get_special_tokens_map(clip)
        chunks = text.split(self.SEPARATOR)

        cond_concat = None
        for chunk in chunks:
            chunk = chunk.strip()
            tokens: dict[str, list[list[tuple[int, float]]]] = clip.tokenize(chunk)
            tokens_inverted: dict[str, list[list[tuple[int, float]]]] = {}
            for clip_name, sections in tokens.items():
                special_tokens = special_tokens_map.get(clip_name, set())
                tokens_inverted[clip_name] = []
                for section in sections:
                    tokens_inverted[clip_name].append([
                        (token, weight if token in special_tokens and not invert_special_tokens else -weight)
                        for token, weight in section
                    ])

            cond, pooled = clip.encode_from_tokens(tokens_inverted, return_pooled=True)
            conditioning = [[cond, {"pooled_output": pooled}]]
            cond_concat = concat_node.concat(cond_concat, conditioning)[0] if cond_concat else conditioning

        return (cond_concat,)


NODE_CLASS_MAPPINGS = {
    "CLIPTextEncodeBREAK": CLIPTextEncodeBREAK,
    "CLIPMicroConditioning": CLIPMicroConditioning,
    "CLIPTokenCounter": CLIPTokenCounter,
    "ConditioningZeroOutCombine": ConditioningZeroOutCombine,
    "CLIPTextEncodeInvertWeights": CLIPTextEncodeInvertWeights,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPTextEncodeBREAK": "CLIP Text Encode (BREAK)",
    "CLIPMicroConditioning": "CLIPMicroConditioning",
    "CLIPTokenCounter": "CLIPTokenCounter",
    "ConditioningZeroOutCombine": "ConditioningZeroOut (Combine)",
    "CLIPTextEncodeInvertWeights": "CLIP Text Encode (Invert Weights)",
}
