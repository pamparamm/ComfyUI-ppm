import math

from comfy.sd import CLIP
from nodes import ConditioningConcat, MAX_RESOLUTION
from node_helpers import conditioning_set_values


class CLIPTextEncodeBREAK:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "clip": ("CLIP",),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
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


class CLIPMicroConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cond": ("CONDITIONING",),
                "width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
                "height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
                "crop_w": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
                "crop_h": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
                "target_width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
                "target_height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "micro_conditioning"

    CATEGORY = "conditioning"

    def micro_conditioning(self, cond, width, height, crop_w, crop_h, target_width, target_height):
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


class CLIPTokenCounter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "clip": ("CLIP",),
                "debug_print": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "encode"

    OUTPUT_NODE = True

    def encode(self, clip: CLIP, text: str, debug_print: bool):
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
                print(f"Token count: {' + '.join(lengths)}")
                print("--start--")
                print(" + ".join(f"'{vocab[t[0]]}'" for b in blocks for t in b))
                print("--finish--")
        else:
            lengths = ["0"]
        return (lengths,)
