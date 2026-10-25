import logging
import math
from typing import Any, Callable

import torch

import comfy.utils
from comfy.sd import CLIP
from comfy.text_encoders.moss_image import MOSSImageTokenizer
from comfy_api.latest import io
from comfy_extras.nodes_moss_image import EDIT_FORMATS

from ..negpip.moss_negpip import entry_rows

EDIT_KEY = "ppm_moss_edit"


def downscale(image: torch.Tensor) -> torch.Tensor:
    scale = min(1.0, math.sqrt(384 * 384 / (image.shape[1] * image.shape[2])))
    return comfy.utils.common_upscale(
        image.movedim(-1, 1), round(image.shape[2] * scale), round(image.shape[1] * scale), "area", "disabled"
    ).movedim(1, -1)


class ImagesTokenizer(MOSSImageTokenizer):
    ppm_blocks: str
    ppm_images: list[torch.Tensor]

    def tokenize_with_weights(self, text: str, return_word_ids: bool = False, **kwargs):  # pyrefly: ignore[bad-override]
        if not kwargs.get("images") and kwargs.get("image") is None:
            kwargs["images"] = self.ppm_images
            text = self.ppm_blocks + text
        return super().tokenize_with_weights(text, return_word_ids=return_word_ids, **kwargs)


def images_tokenizer(tokenizer: MOSSImageTokenizer, images: list[torch.Tensor]) -> ImagesTokenizer:
    blocks = "".join(
        "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1) for i in range(len(images))
    )
    tok = ImagesTokenizer.__new__(ImagesTokenizer)
    tok.__dict__ = {**tokenizer.__dict__, "ppm_blocks": blocks, "ppm_images": images}
    return tok


def images_encode_token_weights(
    previous_encode_token_weights: Callable, weight: float, refs: list[torch.Tensor], method: str
):
    def _images_encode_token_weights(token_weight_pairs):
        out = previous_encode_token_weights(token_weight_pairs)
        extra = out[2] if len(out) > 2 else {}

        bias: list[float] = []
        for section in token_weight_pairs:
            for token in section:
                bias += [weight if isinstance(token[0], dict) else 0.0] * entry_rows(token[0])
        if any(bias):
            extra["context_attention_bias"] = torch.tensor([bias])
        if len(refs) > 0:
            extra["reference_latents"] = refs
            extra["reference_latents_method"] = method

        return (*out[:2], extra)

    return _images_encode_token_weights


def patch_clip(clip: CLIP, images: list[torch.Tensor] | None, weight: float, refs: list[torch.Tensor], method: str):
    encoder = clip.tokenizer.clip_name
    if images is not None:
        clip.tokenizer = images_tokenizer(clip.tokenizer, images)
    clip.patcher.add_object_patch(
        f"{encoder}.encode_token_weights",
        images_encode_token_weights(
            clip.patcher.get_model_object(f"{encoder}.encode_token_weights"), weight, refs, method
        ),
    )


class MOSSImageEditPPM(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MOSSImageEditPPM",
            display_name="MOSS-Image Edit (PPM)",
            category="model/conditioning/moss_image",
            description="Text Encode MOSS-Image Edit, but in a form of a CLIP patch. Unlike the vanilla node, this one is compatible with custom text encode nodes.",
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input(
                    "vae",
                    optional=True,
                    tooltip="Encodes the images as reference latents for the edit LoRA.",
                ),
                io.Combo.Input(
                    "edit_format",
                    options=list(EDIT_FORMATS),
                    tooltip="The layout the edit LoRA was trained with: ai-toolkit's MOSS-Image Edit Training (its kv_cache option on or off) or Moss-Image-Edit. With kv_cache off the references also attend to the image being generated, so they are recomputed every step.",
                ),
                io.Autogrow.Input(
                    "images",
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("image"),
                        prefix="image_",
                        min=1,
                    ),
                ),
                io.Float.Input(
                    "image_strength",
                    default=1.0,
                    min=0.0,
                    max=100.0,
                    step=0.05,
                    advanced=True,
                    tooltip="Scales how much the model attends to the images the text encoder sees (ai-toolkit formats, or no VAE): 0 ignores them, 1 is unchanged, above 1 emphasizes them.",
                ),
            ],
            outputs=[io.Clip.Output()],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        clip: CLIP = kwargs["clip"]
        edit_format: str = kwargs["edit_format"]
        image_strength: float = kwargs["image_strength"]
        vae = kwargs.get("vae")
        images_in: dict = kwargs["images"]

        c: CLIP = clip.clone()
        model_options: dict[str, Any] = c.patcher.model_options

        if not isinstance(c.tokenizer, MOSSImageTokenizer):
            logging.warning(
                "MOSS-Image Edit (PPM): `{}` CLIP cannot take edit references, doing nothing.".format(
                    type(c.tokenizer).__name__
                )
            )
            return io.NodeOutput(c)
        if model_options.get(EDIT_KEY):
            logging.warning("MOSS-Image Edit (PPM): the CLIP already has edit references, keeping the first ones.")
            return io.NodeOutput(c)

        method, vision = EDIT_FORMATS[edit_format]
        images = [images_in[name] for name in sorted(images_in) if images_in[name] is not None]
        refs = [vae.encode(image[:, :, :, :3]) for image in images] if vae is not None else []

        is_enabled = (vision or len(refs) == 0) and image_strength > 0.0
        if len(images) == 0 or (not is_enabled and len(refs) == 0):
            return io.NodeOutput(c)

        model_options[EDIT_KEY] = True
        patch_clip(
            c,
            [downscale(image) for image in images] if is_enabled else None,
            math.log(image_strength) if is_enabled else 0.0,
            refs,
            method,
        )
        return io.NodeOutput(c)


NODES = [MOSSImageEditPPM]
