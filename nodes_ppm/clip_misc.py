import logging

from comfy.comfy_types.node_typing import IO, ComfyNodeABC, InputTypeDict
from comfy.sd import CLIP
from node_helpers import conditioning_set_values
from nodes import (
    MAX_RESOLUTION,
    ConditioningCombine,
    ConditioningConcat,
    ConditioningSetTimestepRange,
    ConditioningZeroOut,
)


class CLIPTextEncodeBREAK(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "text": (IO.STRING, {"multiline": True, "dynamicPrompts": True}),
                "clip": (IO.CLIP, {}),
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
                "text": (IO.STRING, {"multiline": True}),
                "clip": (IO.CLIP, {}),
                "debug_print": (IO.BOOLEAN, {"default": False}),
            }
        }

    RETURN_TYPES = (IO.STRING,)
    FUNCTION = "count"

    OUTPUT_NODE = True

    def count(self, clip: CLIP, text: str, debug_print: bool):
        lengths = []
        blocks = []
        special_tokens = set(clip.cond_stage_model.clip_l.special_tokens.values())
        vocab = clip.tokenizer.clip_l.inv_vocab
        prompts = text.split("BREAK")
        for prompt in prompts:
            if len(prompt) > 0:
                tokens_pure = clip.tokenize(prompt)
                tokens_concat = sum(tokens_pure["l"], [])
                block = [t for t in tokens_concat if t[0] not in special_tokens]
                blocks.append(block)

        if len(blocks) > 0:
            lengths = [str(len(b)) for b in blocks]
            if debug_print:
                logging.info(f"Token count: {' + '.join(lengths)}")
                logging.info("--start--")
                logging.info(" + ".join(f"'{vocab[token[0]]}'" for block in blocks for token in block))
                logging.info("--finish--")
        else:
            lengths = ["0"]
        return (lengths,)


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
