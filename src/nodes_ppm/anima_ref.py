# Anima Reference Adapter by ElectronGirl, licensed under Apache-2.0; adapter LoRA: CircleStone Non-Commercial
# https://github.com/ElectronGirl/ComfyUI-Anima-Ref-Adapter
# https://huggingface.co/ElectronGirl/anima-ref-adapter
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image

import comfy.conds
import comfy.patcher_extension
import comfy.utils
import node_helpers
from comfy.ldm.cosmos.position_embedding import VideoRopePosition3DEmb
from comfy.model_patcher import ModelPatcher
from comfy.model_sampling import ModelSamplingDiscrete
from comfy.sd import VAE
from comfy_api.latest import io

WRAPPER_KEY = "ppm_anima_ref"
REF_LATENTS_KEY = "ppm_anima_ref_latents"
REF_COND_KEY = "c_ppm_anima_ref_latents"
EXTRA_CONDS_KEY = "ppm_anima_ref_inner"


def anima_ref_extra_conds_wrapper(
    previous_extra_conds: Callable[..., dict], process_latent_in: Callable[[torch.Tensor], torch.Tensor]
) -> Callable[..., dict]:
    # a node re-run sees its own patch, so always wrap the original function
    previous_extra_conds = getattr(previous_extra_conds, EXTRA_CONDS_KEY, previous_extra_conds)

    def _anima_ref_extra_conds_wrapper(**kwargs):
        out = previous_extra_conds(**kwargs)
        ref_latents = kwargs.get(REF_LATENTS_KEY)
        if ref_latents is not None:
            out[REF_COND_KEY] = comfy.conds.CONDRegular(process_latent_in(ref_latents[-1]))
        return out

    setattr(_anima_ref_extra_conds_wrapper, EXTRA_CONDS_KEY, previous_extra_conds)
    return _anima_ref_extra_conds_wrapper


def anima_ref_rope_wrapper(
    previous_forward: Callable[..., torch.Tensor], pos_embedder: VideoRopePosition3DEmb, offset_ratio: float
) -> Callable[..., torch.Tensor]:
    previous_forward = getattr(previous_forward, EXTRA_CONDS_KEY, previous_forward)

    def _anima_ref_rope_wrapper(x_B_T_H_W_C: torch.Tensor, fps=None, device=None, dtype=None) -> torch.Tensor:
        emb = previous_forward(x_B_T_H_W_C, fps=fps, device=device, dtype=dtype)
        _, num_frames, height, width, _ = x_B_T_H_W_C.shape
        if num_frames != 2:
            return emb
        # layout: (t h w) (t_dim, h_dim, w_dim) (cos/sin pair) (cos/sin)
        temporal_dim = pos_embedder.dim_temporal_range.numel()
        height_dim = pos_embedder.dim_spatial_range.numel()
        freqs = 1.0 / ((10000.0 * pos_embedder.h_ntk_factor) ** pos_embedder.dim_spatial_range.to(x_B_T_H_W_C.device))
        angle = torch.outer(
            torch.arange(height, device=x_B_T_H_W_C.device, dtype=torch.float) + round(offset_ratio * height), freqs
        )
        shifted = torch.stack([angle.cos(), -angle.sin(), angle.sin(), angle.cos()], dim=-1)
        shifted = shifted.unflatten(-1, (2, 2)).unsqueeze(1).expand(height, width, height_dim, 2, 2)

        target_emb, ref_emb = emb[: height * width], emb[height * width :].reshape(height, width, *emb.shape[1:])
        ref_emb = torch.cat([ref_emb[:, :, :temporal_dim], shifted, ref_emb[:, :, temporal_dim + height_dim :]], dim=2)
        return torch.cat([target_emb, ref_emb.flatten(0, 1)], dim=0)

    setattr(_anima_ref_rope_wrapper, EXTRA_CONDS_KEY, previous_forward)
    return _anima_ref_rope_wrapper


def anima_ref_diffusion_wrapper(executor, x, timesteps, context, fps=None, padding_mask=None, **kwargs):
    ref: torch.Tensor | None = kwargs.pop(REF_COND_KEY, None)
    if ref is None:
        return executor(x, timesteps, context, fps, padding_mask, **kwargs)
    if ref.ndim == 4:
        ref = ref.unsqueeze(2)
    if ref.shape[-2:] != x.shape[-2:]:
        ref = comfy.utils.common_upscale(
            ref[:, :, 0], x.shape[-1], x.shape[-2], "area" if ref.shape[-2] > x.shape[-2] else "bilinear", "disabled"
        )
        ref = ref.unsqueeze(2)
    timestep = timesteps.reshape(x.shape[0], -1)[:, :1]
    x = torch.cat([x, ref], dim=2)
    timesteps = torch.cat([timestep, torch.zeros_like(timestep)], dim=1)
    return executor(x, timesteps, context, fps, padding_mask, **kwargs)[:, :, :1]


def without_ref_conds(conds: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if conds is None:
        return None
    return [{**c, "model_conds": {k: v for k, v in c["model_conds"].items() if k != REF_COND_KEY}} for c in conds]


def anima_ref_calc_cond_batch_wrapper(
    start_sigma: float, end_sigma: float, windowed: bool
) -> Callable[..., list[torch.Tensor]]:
    def _anima_ref_calc_cond_batch_wrapper(
        executor, model, conds: list[list[dict[str, Any]] | None], x_in, timestep, model_options: dict[str, Any]
    ):
        if windowed and not end_sigma <= float(timestep[0]) <= start_sigma:
            return executor(model, [without_ref_conds(c) for c in conds], x_in, timestep, model_options)
        neg_conds = conds[1] if len(conds) == 2 else None
        if neg_conds is None or not any(REF_COND_KEY in c["model_conds"] for c in neg_conds):
            return executor(model, conds, x_in, timestep, model_options)
        return executor(model, conds + [without_ref_conds(neg_conds)], x_in, timestep, model_options)

    return _anima_ref_calc_cond_batch_wrapper


def ref_guidance_pre_cfg_wrapper(ref_guidance: float) -> Callable[[dict[str, Any]], list[torch.Tensor]]:
    def _ref_guidance_pre_cfg(args: dict[str, Any]) -> list[torch.Tensor]:
        out = args["conds_out"]
        if len(out) < 3:
            return out
        # shifting both branches by the same vector keeps cond - uncond alone
        ref_effect = (ref_guidance - 1.0) * (out[1] - out[2])
        return [out[0] + ref_effect, out[1] + ref_effect] + out[2:]

    return _ref_guidance_pre_cfg


class AnimaRefAdapterPPM(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AnimaRefAdapterPPM",
            display_name="Anima Ref Adapter (PPM)",
            category="model/patch/anima",
            search_aliases=["anima reference", "reference image", "character reference"],
            description="Requires loading reference adapter LoRA before this node.",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input(
                    "ref_offset_h",
                    default=1.0,
                    min=0.0,
                    max=4.0,
                    step=0.05,
                    tooltip="RoPE shift of the reference frame in target heights. Reference adapter uses 1.0.",
                ),
                io.Float.Input(
                    "ref_guidance",
                    default=4.5,
                    min=0.0,
                    max=20.0,
                    step=0.1,
                    tooltip="When above 1.0, negative prompt is sampled a second time without the reference, increasing reference strength.",
                ),
                io.Float.Input(
                    "start_percent",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                ),
                io.Float.Input(
                    "end_percent",
                    default=0.8,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                ),
            ],
            outputs=[
                io.Model.Output(),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        model: ModelPatcher = kwargs["model"]
        ref_offset_h: float = kwargs["ref_offset_h"]
        ref_guidance: float = kwargs["ref_guidance"]
        start_percent: float = kwargs["start_percent"]
        end_percent: float = kwargs["end_percent"]
        windowed = start_percent > 0.0 or end_percent < 1.0

        m: ModelPatcher = model.clone()
        m.add_object_patch(
            "extra_conds",
            anima_ref_extra_conds_wrapper(m.get_model_object("extra_conds"), m.get_model_object("process_latent_in")),
        )
        m.add_object_patch(
            "diffusion_model.pos_embedder.forward",
            anima_ref_rope_wrapper(
                m.get_model_object("diffusion_model.pos_embedder.forward"),
                m.get_model_object("diffusion_model.pos_embedder"),  # type: ignore
                ref_offset_h,
            ),
        )
        m.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY, anima_ref_diffusion_wrapper
        )
        if ref_guidance != 1.0 or windowed:
            sampling: ModelSamplingDiscrete = m.get_model_object("model_sampling")  # type: ignore
            m.add_wrapper_with_key(
                comfy.patcher_extension.WrappersMP.CALC_COND_BATCH,
                WRAPPER_KEY,
                anima_ref_calc_cond_batch_wrapper(
                    sampling.percent_to_sigma(start_percent), sampling.percent_to_sigma(end_percent), windowed
                ),
            )
        if ref_guidance != 1.0:
            m.set_model_sampler_pre_cfg_function(
                ref_guidance_pre_cfg_wrapper(ref_guidance), disable_cfg1_optimization=True
            )
        return io.NodeOutput(m)


class AnimaRefConditioningPPM(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AnimaRefConditioningPPM",
            display_name="Anima Ref Conditioning (PPM)",
            category="model/conditioning/anima",
            search_aliases=["anima reference", "reference image", "character reference"],
            description="Pads reference image to latent size and encodes it with VAE.",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Image.Input("image", tooltip="Reference image."),
                io.Latent.Input("latent", tooltip="Latent used during sampling."),
                io.Int.Input("fill", default=255, min=0, max=255, advanced=True),
            ],
            outputs=[
                io.Conditioning.Output(id="positive", display_name="positive"),
                io.Conditioning.Output(id="negative", display_name="negative"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        vae: VAE = kwargs["vae"]
        image: torch.Tensor = kwargs["image"]
        fill: int = kwargs["fill"]

        ratio = vae.spacial_compression_encode()
        height: int = kwargs["latent"]["samples"].shape[-2] * ratio
        width: int = kwargs["latent"]["samples"].shape[-1] * ratio

        src = Image.fromarray((image[0].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8))
        scale = min(width / src.width, height / src.height)
        size = (max(1, round(src.width * scale)), max(1, round(src.height * scale)))
        canvas = Image.new("RGB", (width, height), (fill, fill, fill))
        canvas.paste(src.resize(size, Image.Resampling.LANCZOS), ((width - size[0]) // 2, (height - size[1]) // 2))
        pixels = torch.from_numpy(np.asarray(canvas, dtype=np.float32) / 255.0).unsqueeze(0)

        ref_values = {REF_LATENTS_KEY: [vae.encode(pixels)]}
        return io.NodeOutput(
            node_helpers.conditioning_set_values(kwargs["positive"], ref_values),
            node_helpers.conditioning_set_values(kwargs["negative"], ref_values),
        )


NODES = [AnimaRefAdapterPPM, AnimaRefConditioningPPM]
