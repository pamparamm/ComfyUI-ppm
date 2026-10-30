# Anima Reference Adapter by ElectronGirl, licensed under Apache-2.0; adapter LoRA: CircleStone Non-Commercial
# https://github.com/ElectronGirl/ComfyUI-Anima-Ref-Adapter
# https://huggingface.co/ElectronGirl/anima-ref-adapter
# SnapRef reference adapters by 0ppxnhximxr, licensed under Apache-2.0; adapter LoRA: CircleStone Non-Commercial
# https://github.com/0ppxnhximxr/SnapRef
# https://huggingface.co/0ppxnhximxr/snapref-v1b
import json
import logging
from enum import Enum
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open

import comfy.conds
import comfy.patcher_extension
import comfy.utils
import folder_paths
import node_helpers
from comfy.ldm.cosmos.position_embedding import VideoRopePosition3DEmb
from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention_masked
from comfy.model_patcher import ModelPatcher
from comfy.model_sampling import ModelSamplingDiscrete
from comfy.sd import VAE
from comfy_api.latest import io

WRAPPER_KEY = "ppm_anima_ref"
REF_LATENTS_KEY = "ppm_anima_ref_latents"
REF_COND_KEY = "c_ppm_anima_ref_latents"
REF_TEXT_KEY = "ppm_anima_ref_text"
REF_TEXT_COND_KEY = "c_ppm_anima_ref_text"
INNER_KEY = "ppm_anima_ref_inner"


class AnimaRefAdapterType(Enum):
    ANIMA_REF = 0
    SNAPREF = 1


class AnimaRefAdapterFile:
    def __init__(self, path: str) -> None:
        self.tensors = comfy.utils.load_torch_file(path, safe_load=True)
        with safe_open(path, "pt") as handle:
            metadata = handle.metadata() or {}
        snapref = self._metadata(metadata, ("snapref", "anima_ref_kv"))
        anima_ref = self._metadata(metadata, ("anima_ref_adapter",))
        if snapref.get("format") == "anima_ref_kv/1":
            self.joint: bool = False
            self.t_index = float(snapref.get("ref_t", 10))
            self.offset_h = 0.0
            self.offset_w = 0.0
            self.centred = True
            self.ref_cross_attn = True
            self.ref_in_uncond = False
            self.modes = list(snapref.get("modes", []))
            self.mode_embed = self.tensors.get("anima_ref_kv.mode_embed")
            self.streams = self._streams()
            self.type = AnimaRefAdapterType.SNAPREF
        elif len(anima_ref) > 0:
            slot = anima_ref.get("slots", {}).get("character", {})
            self.joint: bool = anima_ref.get("arch") != "asym"
            self.t_index = float(slot.get("t", 1))
            self.offset_h = float(slot.get("offset_h", anima_ref.get("ref_offset_h", 1.0)))
            self.offset_w = float(slot.get("offset_w", anima_ref.get("ref_offset_w", 0.0)))
            self.centred = False
            self.ref_cross_attn = self.joint or anima_ref.get("ref_cross_attn", True)
            self.ref_in_uncond = True
            self.modes = []
            self.mode_embed = None
            self.streams = self._streams()
            self.type = AnimaRefAdapterType.ANIMA_REF
        else:
            raise ValueError(f"{path} is not an Anima reference adapter")
        if self.joint and set(self.streams) - {None}:
            raise ValueError(f"{path} combines a joint reference frame with per-pass LoRA streams")

    @staticmethod
    def _metadata(metadata: dict, keys: tuple) -> dict:
        for key in keys:
            if key in metadata:
                return json.loads(metadata[key])
        return {}

    def _streams(self) -> dict[str | None, dict[str, list]]:
        streams: dict[str | None, dict[str, list]] = {}
        for key, down in self.tensors.items():
            if not key.endswith((".lora_down", ".lora_down.weight")):
                continue
            stem = key.removesuffix(".weight").removesuffix(".lora_down")
            if stem.startswith("diffusion_model."):
                stream, module = None, stem[len("diffusion_model.") :]
            else:
                _, stream, module = stem.split(".", 2)
            scale = float(self.tensors[stem + ".alpha"]) / down.shape[0]
            streams.setdefault(stream, {})[module] = [
                self._dequant(key),
                self._dequant(key.replace("lora_down", "lora_up")),
                scale,
            ]
        return streams

    def _dequant(self, key: str) -> torch.Tensor:
        weight = self.tensors[key]
        if key + "_scale" in self.tensors:
            weight = weight.float() * self.tensors[key + "_scale"]
        return weight


def _previous(func: Callable) -> Callable:
    return getattr(func, INNER_KEY, func)


def _wrapped(func: Callable, previous: Callable) -> Callable:
    # a node re-run sees its own patch, so every wrapper unwraps before wrapping again
    setattr(func, INNER_KEY, previous)
    return func


def rope_rotations(position: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    angle = torch.outer(position, freqs)
    return torch.stack([angle.cos(), -angle.sin(), angle.sin(), angle.cos()], dim=-1).unflatten(-1, (2, 2))


def ref_frame_rope(
    pos_embedder: VideoRopePosition3DEmb,
    frame_emb_H_W_D_2_2: torch.Tensor,
    t_index: float,
    h_offset: float,
    w_offset: float,
) -> torch.Tensor:
    height, width = frame_emb_H_W_D_2_2.shape[:2]
    device = frame_emb_H_W_D_2_2.device
    temporal_dim = pos_embedder.dim_temporal_range.numel()
    spatial_dim = pos_embedder.dim_spatial_range.numel()  # shared by height and width
    t_freqs = 1.0 / ((10000.0 * pos_embedder.t_ntk_factor) ** pos_embedder.dim_temporal_range.to(device))
    h_freqs = 1.0 / ((10000.0 * pos_embedder.h_ntk_factor) ** pos_embedder.dim_spatial_range.to(device))
    w_freqs = 1.0 / ((10000.0 * pos_embedder.w_ntk_factor) ** pos_embedder.dim_spatial_range.to(device))
    temporal = rope_rotations(torch.full((1,), float(t_index), device=device, dtype=torch.float), t_freqs)
    shifted_h = rope_rotations(torch.arange(height, device=device, dtype=torch.float) + h_offset, h_freqs)
    shifted_w = rope_rotations(torch.arange(width, device=device, dtype=torch.float) + w_offset, w_freqs)
    return torch.cat(
        [
            temporal.expand(height, width, temporal_dim, 2, 2),
            shifted_h.unsqueeze(1).expand(height, width, spatial_dim, 2, 2),
            shifted_w.unsqueeze(0).expand(height, width, spatial_dim, 2, 2),
        ],
        dim=2,
    )


def anima_ref_extra_conds_wrapper(
    previous_extra_conds: Callable[..., dict], process_latent_in: Callable[[torch.Tensor], torch.Tensor]
) -> Callable[..., dict]:
    previous_extra_conds = _previous(previous_extra_conds)

    def _anima_ref_extra_conds_wrapper(**kwargs):
        out = previous_extra_conds(**kwargs)
        ref: torch.Tensor | None = kwargs.get(REF_LATENTS_KEY)
        if ref is not None:
            out[REF_COND_KEY] = comfy.conds.CONDRegular(process_latent_in(ref))
        ref_text: dict[str, torch.Tensor] | None = kwargs.get(REF_TEXT_KEY)
        if ref_text is not None:
            # reuse the model's own text conditioning, including the t5 token ids some encoders return
            out[REF_TEXT_COND_KEY] = previous_extra_conds(**{**kwargs, **ref_text})["c_crossattn"]
        return out

    return _wrapped(_anima_ref_extra_conds_wrapper, previous_extra_conds)


def anima_ref_joint_rope_wrapper(
    previous_forward: Callable[..., torch.Tensor], pos_embedder: VideoRopePosition3DEmb, adapter: AnimaRefAdapterFile
) -> Callable[..., torch.Tensor]:
    previous_forward = _previous(previous_forward)

    def _anima_ref_joint_rope_wrapper(x_B_T_H_W_C: torch.Tensor, fps=None, device=None, dtype=None) -> torch.Tensor:
        emb = previous_forward(x_B_T_H_W_C, fps=fps, device=device, dtype=dtype)
        _, num_frames, height, width, _ = x_B_T_H_W_C.shape
        if num_frames != 2:
            return emb
        target_emb = emb[: height * width]
        ref_emb = emb[height * width :].reshape(height, width, *emb.shape[1:])
        ref_emb = ref_frame_rope(
            pos_embedder, ref_emb, adapter.t_index, round(adapter.offset_h * height), round(adapter.offset_w * width)
        )
        return torch.cat([target_emb, ref_emb.flatten(0, 1)], dim=0)

    return _wrapped(_anima_ref_joint_rope_wrapper, previous_forward)


def anima_ref_capture_rope_wrapper(
    previous_forward: Callable[..., torch.Tensor], pos_embedder: VideoRopePosition3DEmb, state: dict[str, Any]
) -> Callable[..., torch.Tensor]:
    previous_forward = _previous(previous_forward)

    def _anima_ref_capture_rope_wrapper(x_B_T_H_W_C: torch.Tensor, fps=None, device=None, dtype=None) -> torch.Tensor:
        emb = previous_forward(x_B_T_H_W_C, fps=fps, device=device, dtype=dtype)
        if state["stream"] != "ref":
            return emb
        _, _, height, width, _ = x_B_T_H_W_C.shape
        frame = emb.reshape(height, width, *emb.shape[1:])
        return ref_frame_rope(pos_embedder, frame, *state["ref_rope"]).flatten(0, 1)

    return _wrapped(_anima_ref_capture_rope_wrapper, previous_forward)


def anima_ref_biased_attention(
    attn: torch.nn.Module,
    q_B_S_H_D: torch.Tensor,
    k_B_S_H_D: torch.Tensor,
    v_B_S_H_D: torch.Tensor,
    ref_tokens: int,
    gain: float,
    transformer_options: dict[str, Any],
) -> torch.Tensor:
    q, k, v = (t.permute(0, 2, 1, 3) for t in (q_B_S_H_D, k_B_S_H_D, v_B_S_H_D))
    mask = torch.zeros((1, 1, 1, k.shape[2]), device=q.device, dtype=q.dtype)
    mask[..., -ref_tokens:] = gain
    out = optimized_attention_masked(
        q,
        k,
        v,
        attn.n_heads,
        mask,
        skip_reshape=True,
        preferred_attention=attn.comfy_attention,
        transformer_options=transformer_options,
    )
    return attn.output_dropout(attn.output_proj(out))  # type: ignore


def anima_ref_joint_diffusion_wrapper(state: dict[str, Any]) -> Callable[..., torch.Tensor]:
    def _anima_ref_joint_diffusion_wrapper(executor, x, timesteps, context, fps=None, padding_mask=None, **kwargs):
        ref: torch.Tensor | None = kwargs.pop(REF_COND_KEY, None)
        kwargs.pop(REF_TEXT_COND_KEY, None)
        if ref is None:
            return executor(x, timesteps, context, fps, padding_mask, **kwargs)
        if ref.ndim == 4:
            ref = ref.unsqueeze(2)
        if ref.shape[-2:] != x.shape[-2:]:
            mode = "area" if ref.shape[-2] > x.shape[-2] else "bilinear"
            ref = comfy.utils.common_upscale(ref[:, :, 0], x.shape[-1], x.shape[-2], mode, "disabled").unsqueeze(2)
        timestep = timesteps.reshape(x.shape[0], -1)[:, :1]
        x = torch.cat([x, ref], dim=2)
        timesteps = torch.cat([timestep, torch.zeros_like(timestep)], dim=1)
        state["joint"] = True
        try:
            return executor(x, timesteps, context, fps, padding_mask, **kwargs)[:, :, :1]
        finally:
            state["joint"] = False

    return _anima_ref_joint_diffusion_wrapper


def anima_ref_capture_diffusion_wrapper(
    state: dict[str, Any], adapter: AnimaRefAdapterFile
) -> Callable[..., torch.Tensor]:
    def _anima_ref_capture_diffusion_wrapper(executor, x, timesteps, context, fps=None, padding_mask=None, **kwargs):
        ref: torch.Tensor | None = kwargs.pop(REF_COND_KEY, None)
        ref_text: torch.Tensor | None = kwargs.pop(REF_TEXT_COND_KEY, None)
        outer_stream = state["stream"]
        state["stream"] = outer_stream or "tgt"
        try:
            return _anima_ref_capture_diffusion_inner(
                executor, ref, ref_text, x, timesteps, context, fps, padding_mask, **kwargs
            )
        finally:
            state["stream"] = outer_stream

    def _anima_ref_capture_diffusion_inner(executor, ref, ref_text, x, timesteps, context, fps, padding_mask, **kwargs):
        if ref is None:
            if state["stream"] == "ref" or not state["kv"]:
                return executor(x, timesteps, context, fps, padding_mask, **kwargs)
            state["kv"], captured = [], state["kv"]  # this branch runs without the reference
            try:
                return executor(x, timesteps, context, fps, padding_mask, **kwargs)
            finally:
                state["kv"] = captured
        if state["kv"]:
            # the reference pass is timestep independent, its K/V are reused for the rest of the sample
            return executor(x, timesteps, context, fps, padding_mask, **kwargs)
        if ref.ndim == 4:
            ref = ref.unsqueeze(2)
        if ref.shape[0] > 1 and not adapter.ref_cross_attn:
            # a zeroed cross-attention makes the reference pass text independent, one copy is enough
            ref = ref[:1]
        dm = executor.class_obj
        patch = dm.patch_spatial
        tokens_h, tokens_w = -(-x.shape[-2] // patch), -(-x.shape[-1] // patch)
        ref_h, ref_w = -(-ref.shape[-2] // patch), -(-ref.shape[-1] // patch)
        h_offset = round(adapter.offset_h * tokens_h)
        w_offset = round(adapter.offset_w * tokens_w)
        if adapter.centred:
            h_offset += round((tokens_h - ref_h) / 2)
            w_offset += round((tokens_w - ref_w) / 2)
        state["ref_rope"] = (adapter.t_index, h_offset, w_offset)

        state["kv"] = [None] * len(dm.blocks)
        timestep = torch.zeros(ref.shape[0], dtype=timesteps.dtype, device=timesteps.device)
        state["stream"] = "ref"
        try:
            dm(
                ref,
                timestep,
                ref_text if ref_text is not None else context,
                fps=fps,
                transformer_options=kwargs.get("transformer_options"),
            )
        finally:
            state["stream"] = "tgt"
        return executor(x, timesteps, context, fps, padding_mask, **kwargs)

    return _anima_ref_capture_diffusion_wrapper


def anima_ref_mode_wrapper(
    previous_forward: Callable[..., torch.Tensor], state: dict[str, Any], row: torch.Tensor
) -> Callable[..., torch.Tensor]:
    previous_forward = _previous(previous_forward)

    def _anima_ref_mode_wrapper(t_embedding_B_T_D: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        out = previous_forward(t_embedding_B_T_D, *args, **kwargs)
        if state["stream"] != "ref":
            return out
        return out + row.to(out).view(1, 1, -1)

    return _wrapped(_anima_ref_mode_wrapper, previous_forward)


def anima_ref_lora_wrapper(
    previous_forward: Callable[..., torch.Tensor], state: dict[str, Any], streams: dict[str | None, list]
) -> Callable[..., torch.Tensor]:
    previous_forward = _previous(previous_forward)

    def _anima_ref_lora_wrapper(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        out = previous_forward(x, *args, **kwargs)
        if not state["lora"]:
            return out
        weight = streams.get(state["stream"], streams.get(None))
        if weight is None:
            return out
        if weight[0].device != x.device or weight[0].dtype != x.dtype:
            weight[0], weight[1] = weight[0].to(x.device, x.dtype), weight[1].to(x.device, x.dtype)
        return out + torch.nn.functional.linear(torch.nn.functional.linear(x, weight[0]), weight[1]) * weight[2]

    return _wrapped(_anima_ref_lora_wrapper, previous_forward)


def anima_ref_lora_lifecycle(m: ModelPatcher, state: dict[str, Any], entries: list[list]) -> None:
    loaded = [(entry[0], entry[1]) for entry in entries]

    def on_load(patcher: ModelPatcher, *args) -> None:
        for entry in entries:
            entry[0], entry[1] = entry[0].to(patcher.load_device), entry[1].to(patcher.load_device)

    def on_detach(_: ModelPatcher, *args) -> None:
        state["kv"] = []
        for entry, (down, up) in zip(entries, loaded):
            entry[0], entry[1] = down, up

    m.add_callback_with_key(comfy.patcher_extension.CallbacksMP.ON_LOAD, WRAPPER_KEY, on_load)
    m.add_callback_with_key(comfy.patcher_extension.CallbacksMP.ON_DETACH, WRAPPER_KEY, on_detach)
    m.size = m.model_size() + sum(entry[0].nbytes + entry[1].nbytes for entry in entries)


def anima_ref_kv_wrapper(
    previous_compute_attention: Callable[..., torch.Tensor],
    state: dict[str, Any],
    attn: torch.nn.Module,
    index: int,
    weight: float,
    gain: float,
) -> Callable[..., torch.Tensor]:
    previous_compute_attention = _previous(previous_compute_attention)

    def _anima_ref_kv_wrapper(
        q: AttentionTensorContainer,
        k: AttentionTensorContainer,
        v: AttentionTensorContainer,
        transformer_options: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        if state["stream"] == "ref":
            state["kv"][index] = (k.peek(), v.peek())
            return previous_compute_attention(q, k, v, transformer_options)
        if not state["kv"] or weight == 0.0:
            return previous_compute_attention(q, k, v, transformer_options)
        ref_k, ref_v = state["kv"][index]
        if ref_k.shape[0] != k.peek().shape[0]:
            ref_k = ref_k.expand(k.peek().shape[0], *ref_k.shape[1:])
            ref_v = ref_v.expand(k.peek().shape[0], *ref_v.shape[1:])
        if gain == 0.0:
            mixed_out = previous_compute_attention(
                AttentionTensorContainer(q.peek()),
                AttentionTensorContainer(torch.cat([k.peek(), ref_k], dim=1)),
                AttentionTensorContainer(torch.cat([v.peek(), ref_v], dim=1)),
                transformer_options,
            )
        else:
            mixed_out = anima_ref_biased_attention(
                attn,
                q.peek(),
                torch.cat([k.peek(), ref_k], dim=1),
                torch.cat([v.peek(), ref_v], dim=1),
                ref_k.shape[1],
                gain,
                transformer_options or {},
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

    return _wrapped(_anima_ref_kv_wrapper, previous_compute_attention)


def anima_ref_cross_attn_wrapper(
    previous_compute_attention: Callable[..., torch.Tensor], state: dict[str, Any]
) -> Callable[..., torch.Tensor]:
    previous_compute_attention = _previous(previous_compute_attention)

    def _anima_ref_cross_attn_wrapper(
        q: AttentionTensorContainer,
        k: AttentionTensorContainer,
        v: AttentionTensorContainer,
        transformer_options: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        if state["stream"] != "ref":
            return previous_compute_attention(q, k, v, transformer_options)
        tokens: int = q.peek().shape[1]
        return q.peek().new_zeros((q.peek().shape[0], tokens, v.peek().shape[-2] * v.peek().shape[-1]))

    return _wrapped(_anima_ref_cross_attn_wrapper, previous_compute_attention)


def anima_ref_clear_wrapper(state: dict[str, Any]) -> Callable[..., torch.Tensor]:
    def _anima_ref_clear_wrapper(executor, *args, **kwargs):
        try:
            return executor(*args, **kwargs)
        finally:
            state["kv"] = []

    return _anima_ref_clear_wrapper


def anima_ref_block_wrapper(
    previous_compute_attention: Callable[..., torch.Tensor],
    state: dict[str, Any],
    attn: torch.nn.Module,
    weight: float,
    gain: float,
) -> Callable[..., torch.Tensor]:
    previous_compute_attention = _previous(previous_compute_attention)

    def _anima_ref_block_wrapper(
        q: AttentionTensorContainer,
        k: AttentionTensorContainer,
        v: AttentionTensorContainer,
        transformer_options: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        tokens: int = q.peek().shape[1]
        if not state["joint"]:
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
            if gain == 0.0:
                mixed_out = previous_compute_attention(
                    AttentionTensorContainer(target_q),
                    AttentionTensorContainer(k.peek()),
                    AttentionTensorContainer(v.peek()),
                    transformer_options,
                )
            else:
                mixed_out = anima_ref_biased_attention(
                    attn,
                    target_q,
                    k.peek(),
                    v.peek(),
                    tokens - half,
                    gain,
                    transformer_options or {},
                )
            target_out = mixed_out + (1.0 - weight) * (target_out - mixed_out)
        return torch.cat([target_out, reference_out], dim=1)

    return _wrapped(_anima_ref_block_wrapper, previous_compute_attention)


def anima_ref_calc_cond_batch_wrapper(
    state: dict[str, Any], start_sigma: float, end_sigma: float, windowed: bool, keep_reference: bool, ref_guided: bool
) -> Callable[..., list[torch.Tensor]]:
    def without_ref_conds(conds: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        if conds is None:
            return None
        strip = (REF_COND_KEY, REF_TEXT_COND_KEY)
        return [{**c, "model_conds": {k: v for k, v in c["model_conds"].items() if k not in strip}} for c in conds]

    def _anima_ref_calc_cond_batch_wrapper(
        executor, model, conds: list[list[dict[str, Any]] | None], x_in, timestep, model_options: dict[str, Any]
    ):
        if windowed and not end_sigma <= float(timestep[0]) <= start_sigma:
            state["lora"] = keep_reference
            try:
                if keep_reference:
                    return executor(model, conds, x_in, timestep, model_options)
                return executor(model, [without_ref_conds(c) for c in conds], x_in, timestep, model_options)
            finally:
                state["lora"] = True
        if not ref_guided:
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


class AnimaRefPPM(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AnimaRefPPM",
            display_name="Anima Ref (PPM)",
            category="model/patch/anima",
            search_aliases=["anima reference", "reference image", "character reference", "snapref"],
            description="Applies anima_ref or snapref adapters to model and conds.",
            inputs=[
                io.Model.Input("model"),
                io.Vae.Input("vae"),
                io.Image.Input("image", tooltip="Reference image."),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Clip.Input("clip", optional=True, tooltip="Only for snapref when ref_caption isn't empty."),
                io.Latent.Input(
                    "latent",
                    optional=True,
                    tooltip="Empty Latent used during sampling, only for anima_ref_adapter_pilot.",
                ),
                io.Combo.Input(
                    "lora_name",
                    options=folder_paths.get_filename_list("loras"),
                    tooltip="anima_ref (pilot or v2) or snapref (v1a or v1b) LoRA.",
                ),
                io.Float.Input(
                    "ref_guidance",
                    default=1.0,
                    min=0.0,
                    max=20.0,
                    step=0.1,
                    tooltip="When above 1.0, negative cond is sampled a second time without reference, increasing reference strength.\n"
                    "For snapref, 1.0 should be enough.\n"
                    "For anima_ref, it should be equal to CFG scale.",
                ),
                io.Combo.Input(
                    "mode",
                    options=["character and style", "character"],
                    default="character and style",
                    tooltip="Reference modes, only for snapref adapters.",
                ),
                io.Float.Input(
                    "strength_gain",
                    default=0.0,
                    min=-3.0,
                    max=3.0,
                    step=0.1,
                    advanced=True,
                    tooltip="Extra scaling on reference attention. Positive/negative value may increase/decrease reference strength and precision.",
                ),
                io.Float.Input("start_percent", default=0.0, min=0.0, max=1.0, step=0.01),
                io.Float.Input("end_percent", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Combo.Input(
                    "outside_window",
                    options=["no reference", "reference only"],
                    default="reference only",
                    advanced=True,
                    tooltip="For steps outside start/end percent. Drops reference (with adapter LoRA), or keeps it and only stops extra reference guidance call (equivalent to setting ref_guidance to 1.0).",
                ),
                io.String.Input(
                    "block_list",
                    default="",
                    advanced=True,
                    tooltip="Per-block attention weight of the reference, e.g. '14-26:0.0, 27-39:0.3'. Anima 2B (Base) has 0-27 blocks, Anima 2.9B has 0-39. Decreasing weights of some blocks (i.e. `22:0.0` or `25:0.0` for Anima 2.9B) may reduce style leakage from reference image.",
                ),
                io.String.Input(
                    "ref_caption",
                    default="",
                    placeholder="Optional caption, only for snapref adapters",
                    multiline=True,
                    optional=True,
                ),
            ],
            outputs=[
                io.Model.Output(),
                io.Conditioning.Output(id="positive", display_name="positive"),
                io.Conditioning.Output(id="negative", display_name="negative"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        adapter = AnimaRefAdapterFile(folder_paths.get_full_path_or_raise("loras", kwargs["lora_name"]))
        m: ModelPatcher = kwargs["model"].clone()
        num_blocks: int = len(m.get_model_object("diffusion_model.blocks"))  # type: ignore
        weights: dict[int, float] = cls.parse_ref_blocks(kwargs["block_list"], num_blocks)
        gain: float = kwargs["strength_gain"]
        mode: str = cls.map_mode(adapter, kwargs["mode"])
        state: dict[str, Any] = {
            "stream": None,
            "joint": False,
            "lora": True,
            "kv": [],
            "ref_rope": (adapter.t_index, 0, 0),
        }
        m.add_object_patch(
            "extra_conds",
            anima_ref_extra_conds_wrapper(m.get_model_object("extra_conds"), m.get_model_object("process_latent_in")),
        )
        cls.patch_lora(m, state, adapter)
        if adapter.joint:
            cls.patch_joint(m, state, weights, adapter, num_blocks, gain)
        else:
            cls.patch_asym(m, state, weights, adapter, num_blocks, gain, mode)
        cls.patch_guidance(
            m,
            state,
            kwargs["ref_guidance"],
            kwargs["start_percent"],
            kwargs["end_percent"],
            kwargs["outside_window"] == "reference only",
        )

        values = {
            REF_LATENTS_KEY: kwargs["vae"].encode(
                cls.reference_pixels(adapter, kwargs["image"][:1], kwargs["vae"], kwargs.get("latent"))
            )
        }
        ref_text = cls.ref_text(kwargs.get("clip"), kwargs["ref_caption"])
        if ref_text is not None and adapter.joint:
            logging.warning("Anima Ref (PPM): a joint reference frame shares one text context, ref_caption is ignored")
            ref_text = None
        if ref_text is not None:
            values[REF_TEXT_KEY] = ref_text
        positive = node_helpers.conditioning_set_values(kwargs["positive"], values)
        if adapter.ref_in_uncond or kwargs["ref_guidance"] != 1.0:
            negative = node_helpers.conditioning_set_values(kwargs["negative"], values)
        else:
            negative = kwargs["negative"]
        return io.NodeOutput(m, positive, negative)

    @staticmethod
    def parse_ref_blocks(block_list_str: str, num_blocks: int) -> dict[int, float]:
        weights: dict[int, float] = {}
        for item in block_list_str.split(","):
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
    def map_mode(adapter: AnimaRefAdapterFile, mode: str) -> str:
        if adapter.type == AnimaRefAdapterType.SNAPREF:
            snapref_mode_map = {"character and style": "cs", "character": "char"}
            return snapref_mode_map.get(mode, "cs")
        return "none"

    @staticmethod
    def reference_pixels(
        adapter: AnimaRefAdapterFile, pixels: torch.Tensor, vae: VAE, latent: dict | None
    ) -> torch.Tensor:
        if adapter.joint and latent is not None:
            ratio = vae.spacial_compression_encode()
            src = Image.fromarray((pixels[0].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8))
            return AnimaRefPPM.pad(src, latent["samples"].shape[-1] * ratio, latent["samples"].shape[-2] * ratio)
        trained_ref_megapixels = 1.0
        scale = min(1.0, (trained_ref_megapixels * 1024 * 1024 / (pixels.shape[1] * pixels.shape[2])) ** 0.5)
        width, height = (max(16, int(d * scale) // 16 * 16) for d in (pixels.shape[2], pixels.shape[1]))
        return comfy.utils.common_upscale(pixels.movedim(-1, 1), width, height, "lanczos", "disabled").movedim(1, -1)

    @staticmethod
    def pad(src: Image.Image, width: int, height: int) -> torch.Tensor:
        scale = min(width / src.width, height / src.height)
        size = (max(1, round(src.width * scale)), max(1, round(src.height * scale)))
        canvas = Image.new("RGB", (width, height), (255, 255, 255))
        canvas.paste(src.resize(size, Image.Resampling.LANCZOS), ((width - size[0]) // 2, (height - size[1]) // 2))
        return torch.from_numpy(np.asarray(canvas, dtype=np.float32) / 255.0).unsqueeze(0)

    @staticmethod
    def ref_text(clip, caption: str) -> dict | None:
        if clip is None or not caption.strip():
            return None
        entry = clip.encode_from_tokens_scheduled(clip.tokenize(caption.strip()))[0]
        return {
            "cross_attn": entry[0],
            "t5xxl_ids": entry[1].get("t5xxl_ids"),
            "t5xxl_weights": entry[1].get("t5xxl_weights"),
        }

    @staticmethod
    def patch_lora(m: ModelPatcher, state: dict[str, Any], adapter: AnimaRefAdapterFile) -> None:
        modules = {module for streams in adapter.streams.values() for module in streams}
        for module in modules:
            name = f"diffusion_model.{module}.forward"
            weights = {stream: streams[module] for stream, streams in adapter.streams.items() if module in streams}
            m.add_object_patch(name, anima_ref_lora_wrapper(m.get_model_object(name), state, weights))
        anima_ref_lora_lifecycle(
            m, state, [weight for streams in adapter.streams.values() for weight in streams.values()]
        )

    @staticmethod
    def patch_joint(
        m: ModelPatcher,
        state: dict[str, Any],
        weights: dict[int, float],
        adapter: AnimaRefAdapterFile,
        num_blocks: int,
        gain: float,
    ) -> None:
        m.add_object_patch(
            "diffusion_model.pos_embedder.forward",
            anima_ref_joint_rope_wrapper(
                m.get_model_object("diffusion_model.pos_embedder.forward"),
                m.get_model_object("diffusion_model.pos_embedder"),  # type: ignore
                adapter,
            ),
        )
        m.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY, anima_ref_joint_diffusion_wrapper(state)
        )
        for index in range(num_blocks):
            weight = weights.get(index, 1.0)
            if weight == 1.0 and gain == 0.0:
                continue
            name = f"diffusion_model.blocks.{index}.self_attn.compute_attention"
            attn = m.get_model_object(f"diffusion_model.blocks.{index}.self_attn")
            m.add_object_patch(name, anima_ref_block_wrapper(m.get_model_object(name), state, attn, weight, gain))

    @staticmethod
    def patch_asym(
        m: ModelPatcher,
        state: dict[str, Any],
        weights: dict[int, float],
        adapter: AnimaRefAdapterFile,
        num_blocks: int,
        gain: float,
        mode: str,
    ) -> None:
        if adapter.mode_embed is not None:
            if mode in adapter.modes:
                name = "diffusion_model.t_embedding_norm.forward"
                row = adapter.mode_embed[adapter.modes.index(mode)]
                m.add_object_patch(name, anima_ref_mode_wrapper(m.get_model_object(name), state, row))
            else:
                logging.warning(
                    f"Anima Ref (PPM): adapter has modes {adapter.modes}, '{mode}' is unknown, using the base embedding"
                )
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
            anima_ref_capture_diffusion_wrapper(state, adapter),
        )
        m.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, WRAPPER_KEY, anima_ref_clear_wrapper(state)
        )
        for index in range(num_blocks):
            name = f"diffusion_model.blocks.{index}.self_attn.compute_attention"
            attn = m.get_model_object(f"diffusion_model.blocks.{index}.self_attn")
            m.add_object_patch(
                name, anima_ref_kv_wrapper(m.get_model_object(name), state, attn, index, weights.get(index, 1.0), gain)
            )
            if not adapter.ref_cross_attn:
                name = f"diffusion_model.blocks.{index}.cross_attn.compute_attention"
                m.add_object_patch(name, anima_ref_cross_attn_wrapper(m.get_model_object(name), state))

    @staticmethod
    def patch_guidance(
        m: ModelPatcher,
        state: dict[str, Any],
        ref_guidance: float,
        start_percent: float,
        end_percent: float,
        keep_reference: bool,
    ) -> None:
        windowed = start_percent > 0.0 or end_percent < 1.0
        if ref_guidance != 1.0 or windowed:
            sampling: ModelSamplingDiscrete = m.get_model_object("model_sampling")  # type: ignore
            m.add_wrapper_with_key(
                comfy.patcher_extension.WrappersMP.CALC_COND_BATCH,
                WRAPPER_KEY,
                anima_ref_calc_cond_batch_wrapper(
                    state,
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


NODES = [AnimaRefPPM]
