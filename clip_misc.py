import math

from comfy.sd import CLIP
from nodes import CLIPTextEncode, ConditioningConcat, MAX_RESOLUTION
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

    def encode(self, clip: CLIP, text: str):
        encode_node = CLIPTextEncode()
        concat_node = ConditioningConcat()

        chunks = text.split("BREAK")

        cond_concat = encode_node.encode(clip, chunks[0].strip())[0]
        for chunk in chunks[1:]:
            cond = encode_node.encode(clip, chunk.strip())[0]
            cond_concat = concat_node.concat(cond_concat, cond)[0]

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


class CLIPSetLastLayerFloat:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "stop_at_clip_layer": ("FLOAT", {"default": -1, "min": -24, "max": -1, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "set_last_layer"

    CATEGORY = "conditioning"

    def encode_from_tokens_patched(self, clip: CLIP, stop_at_clip_layer):
        def encode_from_tokens(tokens, return_pooled=False):
            clip.cond_stage_model.reset_clip_options()

            if clip.layer_idx is not None:
                clip.cond_stage_model.set_clip_options({"layer": clip.layer_idx})

            if return_pooled == "unprojected":
                clip.cond_stage_model.set_clip_options({"projected_pooled": False})

            clip.load_model()

            clip_skip = -stop_at_clip_layer

            ceil = math.ceil(clip_skip)
            floor = math.floor(clip_skip)
            alpha = clip_skip - floor

            clip.clip_layer(-ceil)
            cond_ceil, pooled_ceil = clip.cond_stage_model.encode_token_weights(tokens)

            clip.clip_layer(-floor)
            cond_floor, pooled_floor = clip.cond_stage_model.encode_token_weights(tokens)

            cond = cond_ceil * alpha + cond_floor * (1 - alpha)

            if return_pooled:
                pooled = pooled_ceil * alpha + pooled_floor * (1 - alpha)
                pooled.conds_list = pooled_floor.conds_list
                return cond, pooled
            return cond

        return encode_from_tokens

    def set_last_layer(self, clip, stop_at_clip_layer):
        # if clip.patcher.model.clip_name == "l":
        c = clip.clone()
        c.encode_from_tokens = self.encode_from_tokens_patched(clip, stop_at_clip_layer)
        return (c,)
