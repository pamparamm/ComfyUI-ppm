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
from comfy.ldm.modules.attention import AttentionTensorContainer
from comfy.model_patcher import ModelPatcher
from comfy.model_sampling import ModelSamplingDiscrete
from comfy.sd import VAE
from comfy_api.latest import io

WRAPPER_KEY = "ppm_anima_ref"
REF_LATENTS_KEY = "ppm_anima_ref_latents"
REF_COND_KEY = "c_ppm_anima_ref_latents"
EXTRA_CONDS_KEY = "ppm_anima_ref_inner"


def rope_rotations(position: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    angle = torch.outer(position, freqs)
    return torch.stack([angle.cos(), -angle.sin(), angle.sin(), angle.cos()], dim=-1).unflatten(-1, (2, 2))


def anima_ref_extra_conds_wrapper(
    previous_extra_conds: Callable[..., dict], process_latent_in: Callable[[torch.Tensor], torch.Tensor]
) -> Callable[..., dict]:
    # a node re-run sees its own patch, so always wrap the original function
    previous_extra_conds = getattr(previous_extra_conds, EXTRA_CONDS_KEY, previous_extra_conds)

    def _anima_ref_extra_conds_wrapper(**kwargs):
        out = previous_extra_conds(**kwargs)
        ref: torch.Tensor | None = kwargs.get(REF_LATENTS_KEY)
        if ref is not None:
            out[REF_COND_KEY] = comfy.conds.CONDRegular(process_latent_in(ref))
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
        shifted = rope_rotations(
            torch.arange(height, device=x_B_T_H_W_C.device, dtype=torch.float) + round(offset_ratio * height), freqs
        )
        shifted = shifted.unsqueeze(1).expand(height, width, height_dim, 2, 2)

        target_emb, ref_emb = emb[: height * width], emb[height * width :].reshape(height, width, *emb.shape[1:])
        ref_emb = torch.cat([ref_emb[:, :, :temporal_dim], shifted, ref_emb[:, :, temporal_dim + height_dim :]], dim=2)
        return torch.cat([target_emb, ref_emb.flatten(0, 1)], dim=0)

    setattr(_anima_ref_rope_wrapper, EXTRA_CONDS_KEY, previous_forward)
    return _anima_ref_rope_wrapper


def anima_ref_capture_rope_wrapper(
    previous_forward: Callable[..., torch.Tensor], pos_embedder: VideoRopePosition3DEmb, state: dict[str, Any]
) -> Callable[..., torch.Tensor]:
    previous_forward = getattr(previous_forward, EXTRA_CONDS_KEY, previous_forward)

    def _anima_ref_capture_rope_wrapper(x_B_T_H_W_C: torch.Tensor, fps=None, device=None, dtype=None) -> torch.Tensor:
        emb = previous_forward(x_B_T_H_W_C, fps=fps, device=device, dtype=dtype)
        if not state["ref_pass"]:
            return emb
        _, _, height, width, _ = x_B_T_H_W_C.shape
        device = x_B_T_H_W_C.device
        temporal_dim = pos_embedder.dim_temporal_range.numel()
        height_dim = pos_embedder.dim_spatial_range.numel()
        t_freqs = 1.0 / ((10000.0 * pos_embedder.t_ntk_factor) ** pos_embedder.dim_temporal_range.to(device))
        h_freqs = 1.0 / ((10000.0 * pos_embedder.h_ntk_factor) ** pos_embedder.dim_spatial_range.to(device))
        temporal = rope_rotations(torch.ones(1, device=device, dtype=torch.float), t_freqs)
        shifted = rope_rotations(torch.arange(height, device=device, dtype=torch.float) + state["shift"], h_freqs)

        frame = emb.reshape(height, width, *emb.shape[1:])
        frame = torch.cat(
            [
                temporal.expand(height, width, temporal_dim, 2, 2),
                shifted.unsqueeze(1).expand(height, width, height_dim, 2, 2),
                frame[:, :, temporal_dim + height_dim :],
            ],
            dim=2,
        )
        return frame.flatten(0, 1)

    setattr(_anima_ref_capture_rope_wrapper, EXTRA_CONDS_KEY, previous_forward)
    return _anima_ref_capture_rope_wrapper


def anima_ref_joint_diffusion_wrapper(state: dict[str, Any]) -> Callable[..., torch.Tensor]:
    def _anima_ref_joint_diffusion_wrapper(executor, x, timesteps, context, fps=None, padding_mask=None, **kwargs):
        ref: torch.Tensor | None = kwargs.pop(REF_COND_KEY, None)
        if ref is None:
            return executor(x, timesteps, context, fps, padding_mask, **kwargs)
        if ref.ndim == 4:
            ref = ref.unsqueeze(2)
        if ref.shape[-2:] != x.shape[-2:]:
            mode = "area" if ref.shape[-2] > x.shape[-2] else "bilinear"
            ref = comfy.utils.common_upscale(ref[:, :, 0], x.shape[-1], x.shape[-2], mode, "disabled")
            ref = ref.unsqueeze(2)
        timestep = timesteps.reshape(x.shape[0], -1)[:, :1]
        x = torch.cat([x, ref], dim=2)
        timesteps = torch.cat([timestep, torch.zeros_like(timestep)], dim=1)
        state["ref_pass"] = True
        try:
            return executor(x, timesteps, context, fps, padding_mask, **kwargs)[:, :, :1]
        finally:
            state["ref_pass"] = False

    return _anima_ref_joint_diffusion_wrapper


def anima_ref_capture_diffusion_wrapper(state: dict[str, Any], ref_offset_h: float) -> Callable[..., torch.Tensor]:
    def _anima_ref_capture_diffusion_wrapper(executor, x, timesteps, context, fps=None, padding_mask=None, **kwargs):
        ref: torch.Tensor | None = kwargs.pop(REF_COND_KEY, None)
        if ref is None:
            if state["ref_pass"] or not state["kv"]:
                return executor(x, timesteps, context, fps, padding_mask, **kwargs)
            state["kv"], captured = [], state["kv"]  # no reference
            try:
                return executor(x, timesteps, context, fps, padding_mask, **kwargs)
            finally:
                state["kv"] = captured
        if state["kv"]:
            return executor(x, timesteps, context, fps, padding_mask, **kwargs)
        if ref.ndim == 4:
            ref = ref.unsqueeze(2)
        if ref.shape[0] > 1 and torch.equal(ref, ref[:1].expand_as(ref)):
            ref = ref[:1]
        dm = executor.class_obj
        state["shift"] = round(ref_offset_h * -(-x.shape[-2] // dm.patch_spatial))
        state["kv"] = [None] * len(dm.blocks)
        state["ref_pass"] = True
        timestep = torch.zeros(ref.shape[0], dtype=timesteps.dtype, device=timesteps.device)
        try:
            dm(ref, timestep, context, fps=fps, transformer_options=kwargs.get("transformer_options"))
        finally:
            state["ref_pass"] = False
        return executor(x, timesteps, context, fps, padding_mask, **kwargs)

    return _anima_ref_capture_diffusion_wrapper


def anima_ref_kv_wrapper(
    previous_compute_attention: Callable[..., torch.Tensor], state: dict[str, Any], index: int, weight: float
) -> Callable[..., torch.Tensor]:
    previous_compute_attention = getattr(previous_compute_attention, EXTRA_CONDS_KEY, previous_compute_attention)

    def _anima_ref_kv_wrapper(
        q: AttentionTensorContainer,
        k: AttentionTensorContainer,
        v: AttentionTensorContainer,
        transformer_options: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        if state["ref_pass"]:
            state["kv"][index] = (k.peek(), v.peek())
            return previous_compute_attention(q, k, v, transformer_options)
        if not state["kv"] or weight == 0.0:
            return previous_compute_attention(q, k, v, transformer_options)
        ref_k, ref_v = state["kv"][index]
        if ref_k.shape[0] != k.peek().shape[0]:
            ref_k = ref_k.expand(k.peek().shape[0], *ref_k.shape[1:])
            ref_v = ref_v.expand(k.peek().shape[0], *ref_v.shape[1:])
        mixed_out = previous_compute_attention(
            AttentionTensorContainer(q.peek()),
            AttentionTensorContainer(torch.cat([k.peek(), ref_k], dim=1)),
            AttentionTensorContainer(torch.cat([v.peek(), ref_v], dim=1)),
            transformer_options,
        )
        if weight == 1.0:
            return mixed_out
        target_out = previous_compute_attention(
            AttentionTensorContainer(q.peek()),
            AttentionTensorContainer(k.peek()),
            AttentionTensorContainer(v.peek()),
            transformer_options,
        )
        return mixed_out + (1.0 - weight) * (target_out - mixed_out)

    setattr(_anima_ref_kv_wrapper, EXTRA_CONDS_KEY, previous_compute_attention)
    return _anima_ref_kv_wrapper


def anima_ref_cross_attn_wrapper(
    previous_compute_attention: Callable[..., torch.Tensor], state: dict[str, Any]
) -> Callable[..., torch.Tensor]:
    previous_compute_attention = getattr(previous_compute_attention, EXTRA_CONDS_KEY, previous_compute_attention)

    def _anima_ref_cross_attn_wrapper(
        q: AttentionTensorContainer,
        k: AttentionTensorContainer,
        v: AttentionTensorContainer,
        transformer_options: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        if not state["ref_pass"]:
            return previous_compute_attention(q, k, v, transformer_options)
        # batch follows ref stream
        tokens: int = q.peek().shape[1]
        return q.peek().new_zeros((q.peek().shape[0], tokens, v.peek().shape[-2] * v.peek().shape[-1]))

    setattr(_anima_ref_cross_attn_wrapper, EXTRA_CONDS_KEY, previous_compute_attention)
    return _anima_ref_cross_attn_wrapper


def anima_ref_clear_wrapper(state: dict[str, Any]) -> Callable[..., torch.Tensor]:
    def _anima_ref_clear_wrapper(executor, *args, **kwargs):
        try:
            return executor(*args, **kwargs)
        finally:
            state["kv"] = []

    return _anima_ref_clear_wrapper


def anima_ref_block_wrapper(
    previous_compute_attention: Callable[..., torch.Tensor], state: dict[str, Any], weight: float
) -> Callable[..., torch.Tensor]:
    previous_compute_attention = getattr(previous_compute_attention, EXTRA_CONDS_KEY, previous_compute_attention)

    def _anima_ref_block_wrapper(
        q: AttentionTensorContainer,
        k: AttentionTensorContainer,
        v: AttentionTensorContainer,
        transformer_options: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        tokens: int = q.peek().shape[1]
        if not state["ref_pass"]:
            return previous_compute_attention(q, k, v, transformer_options)
        half: int = tokens // 2
        target_q, reference_q = q.peek()[:, :half], q.peek()[:, half:]
        target_k, target_v = k.peek()[:, :half], v.peek()[:, :half]
        reference_out = previous_compute_attention(
            AttentionTensorContainer(reference_q),
            AttentionTensorContainer(k.peek()),
            AttentionTensorContainer(v.peek()),
            transformer_options,
        )
        target_out = previous_compute_attention(
            AttentionTensorContainer(target_q),
            AttentionTensorContainer(target_k),
            AttentionTensorContainer(target_v),
            transformer_options,
        )
        if weight > 0.0:
            mixed_out = previous_compute_attention(
                AttentionTensorContainer(target_q),
                AttentionTensorContainer(k.peek()),
                AttentionTensorContainer(v.peek()),
                transformer_options,
            )
            target_out = mixed_out + (1.0 - weight) * (target_out - mixed_out)
        return torch.cat([target_out, reference_out], dim=1)

    setattr(_anima_ref_block_wrapper, EXTRA_CONDS_KEY, previous_compute_attention)
    return _anima_ref_block_wrapper


def anima_ref_calc_cond_batch_wrapper(
    start_sigma: float, end_sigma: float, windowed: bool, keep_reference: bool, ref_guided: bool
) -> Callable[..., list[torch.Tensor]]:
    def without_ref_conds(conds: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        if conds is None:
            return None
        return [{**c, "model_conds": {k: v for k, v in c["model_conds"].items() if k != REF_COND_KEY}} for c in conds]

    def _anima_ref_calc_cond_batch_wrapper(
        executor, model, conds: list[list[dict[str, Any]] | None], x_in, timestep, model_options: dict[str, Any]
    ):
        if windowed and not end_sigma <= float(timestep[0]) <= start_sigma:
            if keep_reference:
                return executor(model, conds, x_in, timestep, model_options)
            return executor(model, [without_ref_conds(c) for c in conds], x_in, timestep, model_options)
        if not ref_guided:  # without ref_guidance the extra branch is never read
            return executor(model, conds, x_in, timestep, model_options)
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
            description="Requires loading matching reference adapter LoRA before this node.",
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input(
                    "adapter_arch",
                    options=["v1 (joint)", "v2 (asymmetric)"],
                    default="v2 (asymmetric)",
                    tooltip="Use `v2 (asymmetric)` with `anima_ref_adapter_v2`; Use `v1 (joint)` with  `anima_ref_adapter_pilot`.",
                ),
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
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                ),
                io.Combo.Input(
                    "outside_window",
                    options=["no reference", "reference only"],
                    default="reference only",
                    advanced=True,
                    tooltip="For steps outside start/end percent. Drops the reference, or keeps it in both CFG branches and only stop the extra reference guidance call.",
                ),
                io.String.Input(
                    "ref_blocks",
                    default="",
                    advanced=True,
                    tooltip="Per-block attention weight of the reference frame, e.g. '14-26:0.0, 27-39:0.3'. Unlisted blocks keep the full reference, empty disables this feature. Anima base has 0-27 blocks, Anima 2.9B has 0-39. Decreasing weights of some blocks (`22:0.0` or `25:0.0` for Anima 2.9B) can reduce style leakage from reference image.",
                ),
            ],
            outputs=[
                io.Model.Output(),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        m: ModelPatcher = kwargs["model"].clone()
        num_blocks: int = len(m.get_model_object("diffusion_model.blocks"))  # type: ignore
        weights: dict[int, float] = cls.parse_ref_blocks(kwargs["ref_blocks"], num_blocks)
        m.add_object_patch(
            "extra_conds",
            anima_ref_extra_conds_wrapper(m.get_model_object("extra_conds"), m.get_model_object("process_latent_in")),
        )
        if kwargs["adapter_arch"] == "v2 (asymmetric)":
            cls.patch_asym(m, weights, kwargs["ref_offset_h"])
        else:
            cls.patch_joint(m, weights, kwargs["ref_offset_h"])
        cls.patch_guidance(
            m,
            kwargs["ref_guidance"],
            kwargs["start_percent"],
            kwargs["end_percent"],
            kwargs["outside_window"] == "reference only",
        )
        return io.NodeOutput(m)

    @staticmethod
    def parse_ref_blocks(spec: str, num_blocks: int) -> dict[int, float]:
        weights: dict[int, float] = {}
        for item in spec.split(","):
            item = item.strip()
            if not item:
                continue
            try:
                blocks, _, weight = item.partition(":")
                bounds = [int(b) for b in blocks.split("-")]
                start, end = bounds[0], bounds[-1]
                weight = float(weight)
            except ValueError:
                raise ValueError(f"ref_blocks: cannot parse '{item}', expected '<block|start-end>:<weight>'") from None
            if not 0 <= start <= end < num_blocks or not 0.0 <= weight <= 1.0:
                raise ValueError(f"ref_blocks: '{item}' out of range, need 0 <= start <= end < {num_blocks}")
            for index in range(start, end + 1):
                weights[index] = weight
        return weights

    @staticmethod
    def patch_joint(m: ModelPatcher, weights: dict[int, float], ref_offset_h: float) -> None:
        state: dict[str, Any] = {"ref_pass": False}
        m.add_object_patch(
            "diffusion_model.pos_embedder.forward",
            anima_ref_rope_wrapper(
                m.get_model_object("diffusion_model.pos_embedder.forward"),
                m.get_model_object("diffusion_model.pos_embedder"),  # type: ignore
                ref_offset_h,
            ),
        )
        m.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY, anima_ref_joint_diffusion_wrapper(state)
        )
        for index, weight in weights.items():
            if weight == 1.0:
                continue
            name = f"diffusion_model.blocks.{index}.self_attn.compute_attention"
            m.add_object_patch(name, anima_ref_block_wrapper(m.get_model_object(name), state, weight))

    @staticmethod
    def patch_asym(m: ModelPatcher, weights: dict[int, float], ref_offset_h: float) -> None:
        state: dict[str, Any] = {"ref_pass": False, "shift": 0, "kv": []}
        m.add_object_patch(
            "diffusion_model.pos_embedder.forward",
            anima_ref_capture_rope_wrapper(
                m.get_model_object("diffusion_model.pos_embedder.forward"),
                m.get_model_object("diffusion_model.pos_embedder"),  # type: ignore
                state,
            ),
        )
        m.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            WRAPPER_KEY,
            anima_ref_capture_diffusion_wrapper(state, ref_offset_h),
        )
        m.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, WRAPPER_KEY, anima_ref_clear_wrapper(state)
        )
        for index in range(len(m.get_model_object("diffusion_model.blocks"))):  # type: ignore
            name = f"diffusion_model.blocks.{index}.self_attn.compute_attention"
            weight = weights.get(index, 1.0)
            m.add_object_patch(name, anima_ref_kv_wrapper(m.get_model_object(name), state, index, weight))
            name = f"diffusion_model.blocks.{index}.cross_attn.compute_attention"
            m.add_object_patch(name, anima_ref_cross_attn_wrapper(m.get_model_object(name), state))

    @staticmethod
    def patch_guidance(
        m: ModelPatcher, ref_guidance: float, start_percent: float, end_percent: float, keep_reference: bool
    ) -> None:
        windowed = start_percent > 0.0 or end_percent < 1.0
        if ref_guidance != 1.0 or windowed:
            sampling: ModelSamplingDiscrete = m.get_model_object("model_sampling")  # type: ignore
            m.add_wrapper_with_key(
                comfy.patcher_extension.WrappersMP.CALC_COND_BATCH,
                WRAPPER_KEY,
                anima_ref_calc_cond_batch_wrapper(
                    sampling.percent_to_sigma(start_percent),
                    sampling.percent_to_sigma(end_percent),
                    windowed,
                    keep_reference,
                    ref_guidance != 1.0,
                ),
            )
        if ref_guidance != 1.0:
            m.set_model_sampler_pre_cfg_function(
                ref_guidance_pre_cfg_wrapper(ref_guidance), disable_cfg1_optimization=True
            )


class AnimaRefConditioningPPM(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AnimaRefConditioningPPM",
            display_name="Anima Ref Conditioning (PPM)",
            category="model/conditioning/anima",
            search_aliases=["anima reference", "reference image", "character reference"],
            description="Encodes reference image with the VAE, and pads it when pad_image is on.",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Image.Input("image", tooltip="Reference image."),
                io.Boolean.Input("pad_image", default=False, tooltip="Necessary for 'v1 (joint)' arch."),
                io.Latent.Input(
                    "latent", optional=True, tooltip="Latent used during sampling. Used when pad_image is on."
                ),
                io.Int.Input(
                    "fill",
                    default=255,
                    min=0,
                    max=255,
                    tooltip="Grayscale value 0-255. Used when pad_image is on.",
                    advanced=True,
                ),
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
        latent: dict | None = kwargs.get("latent")
        pixels = kwargs["image"][:1]
        if kwargs["pad_image"] and latent is not None:
            ratio = vae.spacial_compression_encode()
            src = Image.fromarray((pixels[0].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8))
            pixels = cls.pad(
                src, latent["samples"].shape[-1] * ratio, latent["samples"].shape[-2] * ratio, kwargs["fill"]
            )
        else:
            pixels = cls.resize(pixels)

        ref_values = {REF_LATENTS_KEY: vae.encode(pixels)}
        return io.NodeOutput(
            node_helpers.conditioning_set_values(kwargs["positive"], ref_values),
            node_helpers.conditioning_set_values(kwargs["negative"], ref_values),
        )

    @staticmethod
    def resize(pixels: torch.Tensor) -> torch.Tensor:
        ref_megapixels = 1.0
        scale = (ref_megapixels * 1024 * 1024 / (pixels.shape[1] * pixels.shape[2])) ** 0.5
        width, height = (max(16, round(d * scale / 16) * 16) for d in (pixels.shape[2], pixels.shape[1]))
        return comfy.utils.common_upscale(pixels.movedim(-1, 1), width, height, "lanczos", "disabled").movedim(1, -1)

    @staticmethod
    def pad(src: Image.Image, width: int, height: int, fill: int) -> torch.Tensor:
        scale = min(width / src.width, height / src.height)
        size = (max(1, round(src.width * scale)), max(1, round(src.height * scale)))
        canvas = Image.new("RGB", (width, height), (fill, fill, fill))
        canvas.paste(src.resize(size, Image.Resampling.LANCZOS), ((width - size[0]) // 2, (height - size[1]) // 2))
        return torch.from_numpy(np.asarray(canvas, dtype=np.float32) / 255.0).unsqueeze(0)


NODES = [AnimaRefAdapterPPM, AnimaRefConditioningPPM]
