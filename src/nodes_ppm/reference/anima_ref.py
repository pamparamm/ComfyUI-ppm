# Anima Reference Adapter by ElectronGirl, licensed under Apache-2.0; adapter LoRA: CircleStone Non-Commercial
# https://github.com/ArcAnima/ComfyUI-Anima-Ref-Adapter
# https://huggingface.co/ElectronGirl/anima-ref-adapter
# SnapRef reference adapters by 0ppxnhximxr, licensed under Apache-2.0; adapter LoRA: CircleStone Non-Commercial
# https://github.com/ArcAnima/SnapRef
# https://huggingface.co/0ppxnhximxr/snapref-v1b
import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, cast

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open

import comfy.conds
import comfy.model_prefetch
import comfy.patcher_extension
import comfy.utils
import folder_paths
import node_helpers
from comfy.ldm.cosmos.position_embedding import VideoRopePosition3DEmb
from comfy.ldm.cosmos.predict2 import Attention, MiniTrainDIT
from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention_masked
from comfy.model_base import BaseModel
from comfy.model_patcher import ModelPatcher
from comfy.model_sampling import ModelSamplingDiscrete
from comfy.sd import CLIP, VAE
from comfy_api.latest import io

from ...negpip.anima_negpip import COND_NEGPIP_MASK_KEY
from .ref_cond import PPMRefCond

WRAPPER_KEY = "ppm_anima_ref"
REF_LATENTS_KEY = "ppm_anima_ref_latents"
REF_COND_KEY = "c_ppm_anima_ref_latents"
REF_TEXT_KEY = "ppm_anima_ref_text"
REF_TEXT_COND_KEY = "c_ppm_anima_ref_text"
REF_BIAS_KEY = "ppm_anima_ref_bias"
REF_BIAS_COND_KEY = "c_ppm_anima_ref_bias"
INNER_KEY = "ppm_anima_ref_inner"
RefKV = tuple[torch.Tensor, torch.Tensor]


@dataclass(slots=True)
class AnimaRefLoRA:
    down: torch.Tensor
    up: torch.Tensor
    scale: float


@dataclass(slots=True)
class AnimaRefBranch:
    modes: tuple[str, ...]
    modules: dict[str, AnimaRefLoRA]


class AnimaRefAdapter:
    def __init__(self, path: str) -> None:
        with safe_open(path, "pt") as handle:
            metadata = handle.metadata() or {}
        snapref: dict[str, Any] = json.loads(metadata.get("anima_ref_kv", "{}"))
        anima_ref: dict[str, Any] = json.loads(metadata.get("anima_ref_adapter", "{}"))
        role_ref = metadata.get("anima_role_ref_kv_version")
        if snapref.get("format") == "anima_ref_kv/1" or role_ref is not None:
            self.joint: bool = False
            self.t_index = float(snapref.get("ref_t", 10))
            self.offset_h = 0.0
            self.offset_w = 0.0
            self.centred: bool = True
            self.ref_cross_attn = True
            self.ref_in_uncond: bool = False
            self.modes: list[str] = list(snapref.get("modes", []))
            self.ref_megapixels = 0.5625 if role_ref is not None else 1.0
        elif len(anima_ref) > 0:
            slot = anima_ref.get("slots", {}).get("character", {})
            self.joint: bool = anima_ref.get("arch") != "asym"
            self.t_index = float(slot.get("t", 1))
            self.offset_h = float(slot.get("offset_h", anima_ref.get("ref_offset_h", 1.0)))
            self.offset_w = float(slot.get("offset_w", anima_ref.get("ref_offset_w", 0.0)))
            self.centred: bool = False
            self.ref_cross_attn: bool = bool(anima_ref.get("ref_cross_attn", True))
            self.ref_in_uncond: bool = True
            self.modes: list[str] = []
            self.ref_megapixels = 1.0
        else:
            raise ValueError(f"{path} is not an Anima reference adapter")
        tensor_dict: dict[str, torch.Tensor] = comfy.utils.load_torch_file(path, safe_load=True)
        self.mode_embed: torch.Tensor | None = tensor_dict.get("anima_ref_kv.mode_embed")
        self.streams = self._streams(path, tensor_dict)
        self.branches = self._branches(path, snapref, tensor_dict)
        self.blocks = {
            int(module.split(".")[1])
            for weights in [*self.streams.values(), *(branch.modules for branch in self.branches.values())]
            for module in weights
            if module.startswith("blocks.")
        }

    def block_map(self, num_blocks: int) -> list[int]:
        if not self.blocks:
            return list(range(num_blocks))
        top = max(self.blocks)
        if top >= num_blocks:
            raise ValueError(f"the adapter covers blocks up to {top}, the model has {num_blocks} blocks")
        inserted = {2, 5, 8, 11, 14, 17, 21, 24, 27, 30, 33, 36}
        base = [block for block in range(num_blocks) if block not in inserted]
        if (num_blocks, len(base)) == (40, 28) and top < len(base):
            return base
        return list(range(num_blocks))

    @staticmethod
    def _weight(tensor_dict: dict[str, torch.Tensor], key: str) -> torch.Tensor:
        weight, scale = tensor_dict[key], tensor_dict.get(key + "_scale")
        return weight if scale is None else (weight.float() * scale).to(torch.bfloat16)

    @staticmethod
    def _streams(path: str, tensor_dict: dict[str, torch.Tensor]) -> dict[str | None, dict[str, AnimaRefLoRA]]:
        streams: dict[str | None, dict[str, AnimaRefLoRA]] = {}
        for key in tensor_dict:
            if not key.endswith((".lora_down", ".lora_down.weight")):
                continue
            stem = key.removesuffix(".weight").removesuffix(".lora_down")
            if stem.startswith("diffusion_model."):
                stream, module = None, stem[len("diffusion_model.") :]
            else:
                _, stream, module = stem.split(".", 2)
            down = AnimaRefAdapter._weight(tensor_dict, key)
            up = AnimaRefAdapter._weight(tensor_dict, key.replace("lora_down", "lora_up"))
            streams.setdefault(stream, {})[module] = AnimaRefLoRA(
                down, up, float(tensor_dict[stem + ".alpha"]) / down.shape[0]
            )
        if not streams:
            raise ValueError(f"{path} is not an Anima reference adapter")
        return streams

    @staticmethod
    def _branches(
        path: str, metadata: dict[str, Any], tensor_dict: dict[str, torch.Tensor]
    ) -> dict[str, AnimaRefBranch]:
        prefix = "anima_ref_kv.branch."
        found: dict[str, dict[str, AnimaRefLoRA]] = {}
        for key in tensor_dict:
            if not key.startswith(prefix) or not key.endswith(".down"):
                continue
            stem = key[: -len(".down")]
            name, module = stem[len(prefix) :].split(".", 1)
            down = AnimaRefAdapter._weight(tensor_dict, stem + ".down")
            up = AnimaRefAdapter._weight(tensor_dict, stem + ".up")
            found.setdefault(name, {})[module] = AnimaRefLoRA(
                down, up, float(tensor_dict[stem + ".alpha"]) / down.shape[0]
            )
        branches: dict[str, AnimaRefBranch] = {}
        for name, spec in (metadata.get("branches") or {}).items():
            if name not in found:
                continue
            if spec.get("stream") != "ref" or spec.get("combine", "add") != "add":
                raise ValueError(f"{path} branch {name} needs the ref stream and the add combine")
            branches[name] = AnimaRefBranch(tuple(spec.get("modes") or ()), found.pop(name))
        if found:
            raise ValueError(f"{path} has {', '.join(found)} branch weights without a branches metadata entry")
        return branches

    def get_branches(self, mode: str) -> list[AnimaRefBranch]:
        return [branch for branch in self.branches.values() if mode in branch.modes]


@dataclass(slots=True)
class AnimaRefState:
    biases: tuple[float, ...] = ()
    sizes: list[int] = field(default_factory=list)
    kv: dict[int, RefKV] = field(default_factory=dict)
    span: tuple[int, int] = (0, 0)
    stream: Literal["tgt", "ref"] = "tgt"
    joint: bool = False
    use_lora: bool = True
    ref_rope: tuple[float, int, int] = (0.0, 0, 0)

    @property
    def biased(self) -> bool:
        return any(bias != 0.0 for bias in self.biases)

    def capture(self, index: int, k: torch.Tensor, v: torch.Tensor) -> None:
        if index not in self.kv:
            total = sum(self.sizes)
            with comfy.model_prefetch.pause_malloc_graph():
                self.kv[index] = (
                    k.new_empty((k.shape[0], total, *k.shape[2:])),
                    v.new_empty((v.shape[0], total, *v.shape[2:])),
                )
        start, end = self.span
        self.kv[index][0][:, start:end].copy_(k)
        self.kv[index][1][:, start:end].copy_(v)


def _previous(func: Callable) -> Callable:
    return getattr(func, INNER_KEY, func)


def _wrapped(func: Callable, previous: Callable) -> Callable:
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
    spatial_dim = pos_embedder.dim_spatial_range.numel()
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


def strip_ref_conds(conds: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if conds is None:
        return None
    strip = (REF_COND_KEY, REF_TEXT_COND_KEY, REF_BIAS_COND_KEY)
    return [{**c, "model_conds": {k: v for k, v in c["model_conds"].items() if k not in strip}} for c in conds]


def anima_ref_extra_conds_wrapper(
    previous_extra_conds: Callable[..., dict], process_latent_in: Callable[[torch.Tensor], torch.Tensor]
) -> Callable[..., dict]:
    previous_extra_conds = _previous(previous_extra_conds)

    def _anima_ref_extra_conds_wrapper(**kwargs):
        out = previous_extra_conds(**kwargs)
        refs: list[torch.Tensor] | None = kwargs.get(REF_LATENTS_KEY)
        if refs is not None:
            out[REF_COND_KEY] = comfy.conds.CONDList([process_latent_in(ref) for ref in refs])
            biases: tuple[float, ...] | None = kwargs.get(REF_BIAS_KEY)
            if biases is not None:
                out[REF_BIAS_COND_KEY] = comfy.conds.CONDConstant(tuple(biases))
        ref_texts: list[dict[str, Any] | None] | None = kwargs.get(REF_TEXT_KEY)
        if ref_texts is not None:
            texts = []
            for ref_text in ref_texts:
                if ref_text is None:
                    texts.append(None)
                    continue
                ref_out = previous_extra_conds(**{**kwargs, **ref_text})
                texts.append(ref_out["c_crossattn"].cond)
            out[REF_TEXT_COND_KEY] = comfy.conds.CONDConstant(texts)
        return out

    return _wrapped(_anima_ref_extra_conds_wrapper, previous_extra_conds)


def anima_ref_joint_rope_wrapper(
    previous_forward: Callable[..., torch.Tensor], pos_embedder: VideoRopePosition3DEmb, adapter: AnimaRefAdapter
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
    previous_forward: Callable[..., torch.Tensor], pos_embedder: VideoRopePosition3DEmb, state: AnimaRefState
) -> Callable[..., torch.Tensor]:
    previous_forward = _previous(previous_forward)

    def _anima_ref_capture_rope_wrapper(x_B_T_H_W_C: torch.Tensor, fps=None, device=None, dtype=None) -> torch.Tensor:
        emb = previous_forward(x_B_T_H_W_C, fps=fps, device=device, dtype=dtype)
        if state.stream != "ref":
            return emb
        _, _, height, width, _ = x_B_T_H_W_C.shape
        frame = emb.reshape(height, width, *emb.shape[1:])
        return ref_frame_rope(pos_embedder, frame, *state.ref_rope).flatten(0, 1)

    return _wrapped(_anima_ref_capture_rope_wrapper, previous_forward)


def anima_ref_biased_attention(
    attn: Attention,
    q_B_S_H_D: torch.Tensor,
    k_B_S_H_D: torch.Tensor,
    v_B_S_H_D: torch.Tensor,
    sizes: list[int],
    biases: tuple[float, ...],
    transformer_options: dict[str, Any],
) -> torch.Tensor:
    q, k, v = (t.permute(0, 2, 1, 3) for t in (q_B_S_H_D, k_B_S_H_D, v_B_S_H_D))
    mask = torch.zeros((1, 1, 1, k.shape[2]), device=q.device, dtype=q.dtype)
    start = k.shape[2] - sum(sizes)
    for size, bias in zip(sizes, biases):
        mask[..., start : start + size] = bias
        start += size
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
    return attn.output_dropout(attn.output_proj(out))


def anima_ref_joint_diffusion_wrapper(state: AnimaRefState) -> Callable[..., torch.Tensor]:
    def _anima_ref_joint_diffusion_wrapper(
        executor: comfy.patcher_extension.WrapperExecutor,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        fps: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        refs: list[torch.Tensor] | None = kwargs.pop(REF_COND_KEY, None)
        if refs is None:
            return executor(x, timesteps, context, fps, padding_mask, **kwargs)
        biases: tuple[float, ...] | None = kwargs.pop(REF_BIAS_COND_KEY, None)
        if biases is not None:
            state.biases = biases
        ref = refs[0]
        if ref.ndim == 4:
            ref = ref.unsqueeze(2)
        if ref.shape[-2:] != x.shape[-2:]:
            mode = "area" if ref.shape[-2] > x.shape[-2] else "bilinear"
            ref = comfy.utils.common_upscale(ref[:, :, 0], x.shape[-1], x.shape[-2], mode, "disabled").unsqueeze(2)
        timestep = timesteps.reshape(x.shape[0], -1)[:, :1]
        x = torch.cat([x, ref], dim=2)
        timesteps = torch.cat([timestep, torch.zeros_like(timestep)], dim=1)
        state.joint = True
        try:
            return executor(x, timesteps, context, fps, padding_mask, **kwargs)[:, :, :1]
        finally:
            state.joint = False

    return _anima_ref_joint_diffusion_wrapper


def anima_ref_capture_diffusion_wrapper(state: AnimaRefState, adapter: AnimaRefAdapter) -> Callable[..., torch.Tensor]:
    def capture_reference(
        executor: comfy.patcher_extension.WrapperExecutor,
        refs: list[torch.Tensor],
        ref_texts: list[torch.Tensor | None] | None,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        fps: torch.Tensor | None,
        kwargs: dict[str, Any],
    ) -> None:
        dm = cast(MiniTrainDIT, executor.class_obj)
        patch = dm.patch_spatial
        refs = [ref.unsqueeze(2) if ref.ndim == 4 else ref for ref in refs]
        tokens_h, tokens_w = -(-x.shape[-2] // patch), -(-x.shape[-1] // patch)
        negpip_mask: torch.Tensor | None = kwargs.get(COND_NEGPIP_MASK_KEY)
        state.sizes = [
            -(-ref.shape[2] // dm.patch_temporal) * -(-ref.shape[-2] // patch) * -(-ref.shape[-1] // patch)
            for ref in refs
        ]
        state.kv, start = {}, 0
        for index, ref in enumerate(refs):
            if ref.shape[0] > 1 and not adapter.ref_cross_attn:
                ref = ref[:1]
            ref_h, ref_w = -(-ref.shape[-2] // patch), -(-ref.shape[-1] // patch)
            h_offset = round(adapter.offset_h * tokens_h)
            w_offset = round(adapter.offset_w * tokens_w)
            if adapter.centred:
                h_offset += round((tokens_h - ref_h) / 2)
                w_offset += round((tokens_w - ref_w) / 2)
            state.ref_rope = (adapter.t_index, h_offset, w_offset)
            state.span = (start, start + state.sizes[index])
            start += state.sizes[index]
            timestep = torch.zeros(ref.shape[0], dtype=timesteps.dtype, device=timesteps.device)
            text = context if ref_texts is None or ref_texts[index] is None else ref_texts[index]
            ref_negpip = {COND_NEGPIP_MASK_KEY: negpip_mask[: ref.shape[0]]} if negpip_mask is not None else {}
            state.stream = "ref"
            try:
                dm(ref, timestep, text, fps=fps, transformer_options=kwargs.get("transformer_options"), **ref_negpip)
            finally:
                state.stream = "tgt"

    def _anima_ref_capture_diffusion_wrapper(
        executor: comfy.patcher_extension.WrapperExecutor,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        fps: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        refs: list[torch.Tensor] | None = kwargs.pop(REF_COND_KEY, None)
        ref_texts: list[torch.Tensor | None] | None = kwargs.pop(REF_TEXT_COND_KEY, None)
        biases: tuple[float, ...] | None = kwargs.pop(REF_BIAS_COND_KEY, None)
        if biases is not None:
            state.biases = biases
        if refs is None:
            if state.stream == "ref" or not state.kv:
                return executor(x, timesteps, context, fps, padding_mask, **kwargs)
            state.kv, captured = {}, state.kv
            try:
                return executor(x, timesteps, context, fps, padding_mask, **kwargs)
            finally:
                state.kv = captured
        elif not state.kv:
            capture_reference(executor, refs, ref_texts, x, timesteps, context, fps, kwargs)
        return executor(x, timesteps, context, fps, padding_mask, **kwargs)

    return _anima_ref_capture_diffusion_wrapper


def anima_ref_mode_wrapper(
    previous_forward: Callable[..., torch.Tensor], state: AnimaRefState, row: torch.Tensor
) -> Callable[..., torch.Tensor]:
    previous_forward = _previous(previous_forward)

    def _anima_ref_mode_wrapper(t_embedding_B_T_D: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        out = previous_forward(t_embedding_B_T_D, *args, **kwargs)
        if state.stream != "ref":
            return out
        return out + row.to(out).view(1, 1, -1)

    return _wrapped(_anima_ref_mode_wrapper, previous_forward)


def anima_ref_lora_wrapper(
    previous_forward: Callable[..., torch.Tensor],
    state: AnimaRefState,
    streams: dict[str | None, AnimaRefLoRA],
    strengths: dict[str | None, float],
    branches: list[AnimaRefLoRA],
) -> Callable[..., torch.Tensor]:
    previous_forward = _previous(previous_forward)

    def delta(lora: AnimaRefLoRA, x: torch.Tensor, strength: float) -> torch.Tensor:
        if lora.down.device != x.device or lora.down.dtype != x.dtype:
            lora.down, lora.up = lora.down.to(x.device, x.dtype), lora.up.to(x.device, x.dtype)
        return torch.nn.functional.linear(torch.nn.functional.linear(x, lora.down), lora.up) * (lora.scale * strength)

    def _anima_ref_lora_wrapper(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        out = previous_forward(x, *args, **kwargs)
        if not state.use_lora:
            return out
        key: str | None = state.stream if state.stream in streams else None
        if (lora := streams.get(key)) is not None:
            out = out + delta(lora, x, strengths[key])
        if state.stream == "ref":
            for branch in branches:
                out = out + delta(branch, x, strengths["ref"])
        return out

    return _wrapped(_anima_ref_lora_wrapper, previous_forward)


def anima_ref_lora_lifecycle(m: ModelPatcher, entries: list[AnimaRefLoRA]) -> None:
    resident = [(entry.down, entry.up) for entry in entries]

    def on_load(patcher: ModelPatcher, *args) -> None:
        for entry in entries:
            entry.down, entry.up = entry.down.to(patcher.load_device), entry.up.to(patcher.load_device)

    def on_detach(_: ModelPatcher, *args) -> None:
        for entry, (down, up) in zip(entries, resident):
            entry.down, entry.up = down, up

    m.add_callback_with_key(comfy.patcher_extension.CallbacksMP.ON_LOAD, WRAPPER_KEY, on_load)
    m.add_callback_with_key(comfy.patcher_extension.CallbacksMP.ON_DETACH, WRAPPER_KEY, on_detach)
    m.size = m.model_size() + sum(entry.down.nbytes + entry.up.nbytes for entry in entries)


def anima_ref_kv_wrapper(
    previous_compute_attention: Callable[..., torch.Tensor],
    state: AnimaRefState,
    attn: Attention,
    index: int,
    weight: float,
) -> Callable[..., torch.Tensor]:
    previous_compute_attention = _previous(previous_compute_attention)

    def _anima_ref_kv_wrapper(
        q: AttentionTensorContainer,
        k: AttentionTensorContainer,
        v: AttentionTensorContainer,
        transformer_options: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        if state.stream == "ref":
            if weight != 0.0:
                state.capture(index, k.peek(), v.peek())
            return previous_compute_attention(q, k, v, transformer_options)
        if not state.kv or weight == 0.0:
            return previous_compute_attention(q, k, v, transformer_options)
        ref_k, ref_v = state.kv[index]
        if ref_k.shape[0] != k.peek().shape[0]:
            ref_k = ref_k.expand(k.peek().shape[0], *ref_k.shape[1:])
            ref_v = ref_v.expand(k.peek().shape[0], *ref_v.shape[1:])
        if state.biased:
            mixed_out = anima_ref_biased_attention(
                attn,
                q.peek(),
                torch.cat([k.peek(), ref_k], dim=1),
                torch.cat([v.peek(), ref_v], dim=1),
                state.sizes,
                state.biases,
                transformer_options or {},
            )
        else:
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

    return _wrapped(_anima_ref_kv_wrapper, previous_compute_attention)


def anima_ref_cross_attn_wrapper(
    previous_compute_attention: Callable[..., torch.Tensor], state: AnimaRefState
) -> Callable[..., torch.Tensor]:
    previous_compute_attention = _previous(previous_compute_attention)

    def _anima_ref_cross_attn_wrapper(
        q: AttentionTensorContainer,
        k: AttentionTensorContainer,
        v: AttentionTensorContainer,
        transformer_options: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        if state.stream != "ref":
            return previous_compute_attention(q, k, v, transformer_options)
        tokens: int = q.peek().shape[1]
        return q.peek().new_zeros((q.peek().shape[0], tokens, v.peek().shape[-2] * v.peek().shape[-1]))

    return _wrapped(_anima_ref_cross_attn_wrapper, previous_compute_attention)


def anima_ref_clear_wrapper(state: AnimaRefState) -> Callable[..., torch.Tensor]:
    def _anima_ref_clear_wrapper(executor, *args, **kwargs):
        try:
            return executor(*args, **kwargs)
        finally:
            state.kv = {}

    return _anima_ref_clear_wrapper


def anima_ref_block_wrapper(
    previous_compute_attention: Callable[..., torch.Tensor],
    state: AnimaRefState,
    attn: Attention,
    weight: float,
) -> Callable[..., torch.Tensor]:
    previous_compute_attention = _previous(previous_compute_attention)

    def _anima_ref_block_wrapper(
        q: AttentionTensorContainer,
        k: AttentionTensorContainer,
        v: AttentionTensorContainer,
        transformer_options: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        if weight == 1.0 and not state.biased:
            return previous_compute_attention(q, k, v, transformer_options)
        tokens: int = q.peek().shape[1]
        if not state.joint:
            return previous_compute_attention(q, k, v, transformer_options)
        half: int = tokens // 2
        target_q, reference_q = q.peek()[:, :half], q.peek()[:, half:]
        reference_out = previous_compute_attention(
            AttentionTensorContainer(reference_q),
            AttentionTensorContainer(k.peek()),
            AttentionTensorContainer(v.peek()),
            transformer_options,
        )
        target_out = None
        if weight < 1.0:
            target_out = previous_compute_attention(
                AttentionTensorContainer(target_q),
                AttentionTensorContainer(k.peek()[:, :half]),
                AttentionTensorContainer(v.peek()[:, :half]),
                transformer_options,
            )
        if state.biased:
            mixed_out = anima_ref_biased_attention(
                attn, target_q, k.peek(), v.peek(), [tokens - half], state.biases, transformer_options or {}
            )
        else:
            mixed_out = previous_compute_attention(
                AttentionTensorContainer(target_q),
                AttentionTensorContainer(k.peek()),
                AttentionTensorContainer(v.peek()),
                transformer_options,
            )
        if target_out is not None:
            mixed_out = mixed_out + (1.0 - weight) * (target_out - mixed_out)
        return torch.cat([mixed_out, reference_out], dim=1)

    return _wrapped(_anima_ref_block_wrapper, previous_compute_attention)


def anima_ref_window_wrapper(
    state: AnimaRefState, start_sigma: float, end_sigma: float
) -> Callable[..., list[torch.Tensor]]:
    def _anima_ref_window_wrapper(
        executor: comfy.patcher_extension.WrapperExecutor,
        model: BaseModel,
        conds: list[list[dict[str, Any]] | None],
        x_in: torch.Tensor,
        timestep: torch.Tensor,
        model_options: dict[str, Any],
    ) -> list[torch.Tensor]:
        sigma: float = float(timestep[0])
        if end_sigma <= sigma <= start_sigma:
            return executor(model, conds, x_in, timestep, model_options)
        state.use_lora = False
        try:
            return executor(model, [strip_ref_conds(c) for c in conds], x_in, timestep, model_options)
        finally:
            state.use_lora = True
            state.kv = {}

    return _anima_ref_window_wrapper


class AnimaRefPPM(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AnimaRefPPM",
            display_name="Anima Reference (PPM)",
            category="model/patch/anima",
            search_aliases=["anima reference", "reference image", "character reference", "snapref"],
            description="Applies anima_ref, snapref or anima-role-ref-kv adapters to model and conds.\n"
            "References come from either `Reference Cond (PPM)` or `Reference Gallery (PPM)` nodes, each with its own strength and caption.",
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                PPMRefCond.Input(
                    "ppm_ref_cond",
                    tooltip="Reference from Reference Cond (PPM) or Reference Gallery (PPM) nodes.",
                ),
                io.Latent.Input(
                    "latent",
                    optional=True,
                    tooltip="Empty Latent used during sampling, only for anima_ref_adapter_pilot.",
                ),
                io.Combo.Input(
                    "lora_name",
                    options=folder_paths.get_filename_list("loras"),
                    tooltip="The name of the reference LoRA.\nSupported LoRAs: `anima-ref-adapter`, `snapref`, `anima-role-ref-kv`.",
                ),
                io.Float.Input(
                    "strength_model",
                    default=1.0,
                    min=-100.0,
                    max=100.0,
                    step=0.01,
                    advanced=True,
                    tooltip="How strongly to apply the adapter LoRA weights. Should be set to 1.0 in almost any case.",
                ),
                io.Float.Input(
                    "strength_tgt",
                    default=1.0,
                    min=-100.0,
                    max=100.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Combo.Input(
                    "mode",
                    options=["character and style", "character", "style"],
                    default="character and style",
                    tooltip="Reference modes, only for snapref adapters (correspond to `cs`, `char` and `style` modes).",
                ),
                io.Float.Input(
                    "ref_strength_gain",
                    default=0.0,
                    min=-3.0,
                    max=3.0,
                    step=0.1,
                    advanced=True,
                    tooltip="Extra scaling on reference attention. Positive/negative value may increase/decrease reference strength and precision.\n"
                    "Added on top of the per-reference strengths.",
                ),
                io.Float.Input(
                    "ref_megapixels",
                    default=0.0,
                    min=0.0,
                    max=2.0,
                    step=0.05,
                    advanced=True,
                    tooltip="Reference is shrunk to this resolution.\n0.0 uses the resolution the adapter was trained at.",
                ),
                io.Combo.Input(
                    "multi_layout",
                    options=["separate", "stitch"],
                    default="separate",
                    advanced=True,
                    tooltip="separate: each reference is encoded on its own.\n"
                    "stitch: all references become one image, which is cheaper, but it drops per-reference strengths and captions.",
                ),
                io.Combo.Input(
                    "strength_norm",
                    options=["none", "reference count", "strength sum"],
                    default="reference count",
                    advanced=True,
                    tooltip="none: every reference is used with its strength value unchanged (like how it works in original snapref nodes).\n"
                    "reference count: divide each individual strength by the number of references.\n"
                    "strength sum: divide each individual strength by total strength sum.",
                ),
                io.Float.Input("start_percent", default=0.0, min=0.0, max=1.0, step=0.01),
                io.Float.Input("end_percent", default=1.0, min=0.0, max=1.0, step=0.01),
                io.String.Input(
                    "block_list",
                    default="",
                    advanced=True,
                    tooltip="Per-block attention weight of the reference, e.g. '14-26:0.0, 27-39:0.3'. Anima 2B (Base) has 0-27 blocks, Anima 2.9B has 0-39. Decreasing weights of some blocks (i.e. `22:0.0` or `25:0.0` for Anima 2.9B) may reduce style leakage from reference image.",
                ),
                io.String.Input(
                    "ref_caption",
                    default="",
                    placeholder="Optional appearance caption for all references. References with their own non-empty caption ignore it.",
                    multiline=True,
                    optional=True,
                    tooltip="Optional appearance caption for all references. References with their own non-empty caption ignore it.",
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
        adapter = AnimaRefAdapter(folder_paths.get_full_path_or_raise("loras", kwargs["lora_name"]))
        refs = cls.reference_images(
            kwargs["ppm_ref_cond"], kwargs["ref_caption"], kwargs["multi_layout"], adapter.joint
        )
        if not refs:
            return io.NodeOutput(kwargs["model"], kwargs["positive"], kwargs["negative"])
        m: ModelPatcher = kwargs["model"].clone()
        strengths: list[float] = [strength for _, strength, _ in refs]
        norm: str = kwargs["strength_norm"]
        if norm == "reference count":
            divisor = math.log(len(refs))
        elif norm == "strength sum":
            divisor = math.log(sum(strengths))
        else:
            divisor = 0.0
        gain: float = kwargs["ref_strength_gain"]
        biases = tuple(gain + math.log(strength) - divisor for strength in strengths)
        state = AnimaRefState()
        m.add_object_patch(
            "extra_conds",
            anima_ref_extra_conds_wrapper(m.get_model_object("extra_conds"), m.get_model_object("process_latent_in")),
        )
        m.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, WRAPPER_KEY, anima_ref_clear_wrapper(state)
        )
        modes = {
            "character and style": "cs",
            "character": "char",
            "style": "style",
        }
        mode = modes.get(kwargs["mode"], "cs")
        cls.patch_lora(m, state, adapter, mode, kwargs["strength_model"], kwargs["strength_tgt"])
        if adapter.joint:
            cls.patch_joint(m, state, adapter, kwargs["block_list"])
        else:
            cls.patch_asym(m, state, adapter, kwargs["block_list"], mode)
        cls.patch_window(m, state, kwargs["start_percent"], kwargs["end_percent"])

        vae: VAE = kwargs["vae"]
        values = {
            REF_LATENTS_KEY: [
                vae.encode(cls.reference_pixels(adapter, pixels, vae, kwargs.get("latent"), kwargs["ref_megapixels"]))
                for pixels, _, _ in refs
            ],
            REF_BIAS_KEY: biases,
        }
        ref_text = cls.ref_text(kwargs["clip"], [caption for _, _, caption in refs])
        if ref_text is not None and adapter.joint:
            logging.warning(
                "Anima Reference (PPM): a joint reference frame shares one text context, captions are ignored"
            )
            ref_text = None
        if ref_text is not None:
            values[REF_TEXT_KEY] = ref_text
        positive = node_helpers.conditioning_set_values(kwargs["positive"], values)
        negative = kwargs["negative"]
        if adapter.ref_in_uncond:
            negative = node_helpers.conditioning_set_values(negative, values)
        return io.NodeOutput(m, positive, negative)

    @staticmethod
    def reference_images(
        refs: list[dict[str, Any]], ref_caption: str, multi_layout: str, single: bool
    ) -> list[tuple[torch.Tensor, float, str]]:
        images = [
            (entry["image"][0:1], float(entry.get("strength", 1.0)), str(entry.get("caption", ""))) for entry in refs
        ]
        images = [image for image in images if image[1] > 0.0]
        if single:
            images = images[:1]
        elif multi_layout == "stitch" and len(images) > 1:
            stitched = AnimaRefPPM.stitch([pixels for pixels, _, _ in images])
            return [(stitched, 1.0, ref_caption.strip() or images[0][2].strip())]
        return [(pixels, strength, caption.strip() or ref_caption.strip()) for pixels, strength, caption in images]

    @staticmethod
    def stitch(pixels: list[torch.Tensor]) -> torch.Tensor:
        height = pixels[0].shape[1]
        parts = [
            comfy.utils.common_upscale(
                image.movedim(-1, 1),
                max(1, round(image.shape[2] * height / image.shape[1])),
                height,
                "lanczos",
                "disabled",
            ).movedim(1, -1)
            for image in pixels
        ]
        return torch.cat(parts, dim=2)

    @staticmethod
    def parse_ref_blocks(block_list_str: str, num_blocks: int) -> dict[int, float]:
        weights: dict[int, float] = {}
        for item in block_list_str.split(","):
            item = item.strip()
            if not item:
                continue
            blocks, _, weight = item.partition(":")
            try:
                bounds = [int(b) for b in blocks.split("-")]
                weight = float(weight)
            except ValueError:
                raise ValueError(f"ref_blocks: cannot parse '{item}', expected '<block|start-end>:<weight>'") from None
            for index in range(max(0, bounds[0]), min(num_blocks, bounds[-1] + 1)):
                weights[index] = weight
        return weights

    @staticmethod
    def reference_pixels(
        adapter: AnimaRefAdapter,
        pixels: torch.Tensor,
        vae: VAE,
        latent: dict[str, torch.Tensor] | None,
        megapixels: float,
    ) -> torch.Tensor:
        if adapter.joint and latent is not None:
            ratio = vae.spacial_compression_encode()
            src = Image.fromarray((pixels[0].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8))
            return AnimaRefPPM.pad(src, latent["samples"].shape[-1] * ratio, latent["samples"].shape[-2] * ratio)
        trained = megapixels if megapixels > 0.0 else adapter.ref_megapixels
        scale = min(1.0, (trained * 1024 * 1024 / (pixels.shape[1] * pixels.shape[2])) ** 0.5)
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
    def ref_text(clip: CLIP | None, captions: list[str]) -> list[dict[str, Any] | None] | None:
        if clip is None:
            return None
        encoded: dict[str, dict[str, Any]] = {}
        for caption in captions:
            if caption and caption not in encoded:
                text, extra = clip.encode_from_tokens_scheduled(clip.tokenize(caption))[0]
                encoded[caption] = {"cross_attn": text, **extra}
        texts = [encoded.get(caption) for caption in captions]
        return texts if any(texts) else None

    @staticmethod
    def block_count(m: ModelPatcher) -> int:
        return len(cast(torch.nn.ModuleList, m.get_model_object("diffusion_model.blocks")))

    @staticmethod
    def patch_lora(
        m: ModelPatcher,
        state: AnimaRefState,
        adapter: AnimaRefAdapter,
        mode: str,
        strength_model: float,
        strength_tgt: float,
    ) -> None:
        strengths = {stream: strength_model * (strength_tgt if stream == "tgt" else 1.0) for stream in adapter.streams}
        if not any(strengths.values()):
            return
        blocks = adapter.block_map(AnimaRefPPM.block_count(m))
        modules = {module for stream in adapter.streams.values() for module in stream}
        branches: list[AnimaRefBranch] = adapter.get_branches(mode) if adapter.modes else []
        modules |= {module for branch in branches for module in branch.modules}
        for module in modules:
            if module.startswith("blocks."):
                _, index, tail = module.split(".", 2)
                target = f"blocks.{blocks[int(index)]}.{tail}"
            else:
                target = module
            name = f"diffusion_model.{target}.forward"
            streams = {
                stream: stream_weights[module]
                for stream, stream_weights in adapter.streams.items()
                if module in stream_weights
            }
            branch_loras = [branch.modules[module] for branch in branches if module in branch.modules]
            m.add_object_patch(
                name, anima_ref_lora_wrapper(m.get_model_object(name), state, streams, strengths, branch_loras)
            )
        entries = [lora for stream in adapter.streams.values() for lora in stream.values()]
        entries += [lora for branch in branches for lora in branch.modules.values()]
        anima_ref_lora_lifecycle(m, entries)

    @staticmethod
    def patch_joint(
        m: ModelPatcher,
        state: AnimaRefState,
        adapter: AnimaRefAdapter,
        block_list: str,
    ) -> None:
        num_blocks = AnimaRefPPM.block_count(m)
        weights = AnimaRefPPM.parse_ref_blocks(block_list, num_blocks)
        m.add_object_patch(
            "diffusion_model.pos_embedder.forward",
            anima_ref_joint_rope_wrapper(
                m.get_model_object("diffusion_model.pos_embedder.forward"),
                cast(VideoRopePosition3DEmb, m.get_model_object("diffusion_model.pos_embedder")),
                adapter,
            ),
        )
        m.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY, anima_ref_joint_diffusion_wrapper(state)
        )
        for index in range(num_blocks):
            name = f"diffusion_model.blocks.{index}.self_attn.compute_attention"
            attn = cast(Attention, m.get_model_object(f"diffusion_model.blocks.{index}.self_attn"))
            m.add_object_patch(
                name, anima_ref_block_wrapper(m.get_model_object(name), state, attn, weights.get(index, 1.0))
            )

    @staticmethod
    def patch_asym(
        m: ModelPatcher,
        state: AnimaRefState,
        adapter: AnimaRefAdapter,
        block_list: str,
        mode: str,
    ) -> None:
        num_blocks = AnimaRefPPM.block_count(m)
        weights = AnimaRefPPM.parse_ref_blocks(block_list, num_blocks)
        if adapter.mode_embed is not None:
            if mode in adapter.modes:
                name = "diffusion_model.t_embedding_norm.forward"
                row = adapter.mode_embed[adapter.modes.index(mode)]
                m.add_object_patch(name, anima_ref_mode_wrapper(m.get_model_object(name), state, row))
        m.add_object_patch(
            "diffusion_model.pos_embedder.forward",
            anima_ref_capture_rope_wrapper(
                m.get_model_object("diffusion_model.pos_embedder.forward"),
                cast(VideoRopePosition3DEmb, m.get_model_object("diffusion_model.pos_embedder")),
                state,
            ),
        )
        m.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            WRAPPER_KEY,
            anima_ref_capture_diffusion_wrapper(state, adapter),
        )
        for index in range(num_blocks):
            name = f"diffusion_model.blocks.{index}.self_attn.compute_attention"
            attn = cast(Attention, m.get_model_object(f"diffusion_model.blocks.{index}.self_attn"))
            m.add_object_patch(
                name,
                anima_ref_kv_wrapper(m.get_model_object(name), state, attn, index, weights.get(index, 1.0)),
            )
            if not adapter.ref_cross_attn:
                name = f"diffusion_model.blocks.{index}.cross_attn.compute_attention"
                m.add_object_patch(name, anima_ref_cross_attn_wrapper(m.get_model_object(name), state))

    @staticmethod
    def patch_window(m: ModelPatcher, state: AnimaRefState, start_percent: float, end_percent: float) -> None:
        if start_percent <= 0.0 and end_percent >= 1.0:
            return
        sampling = cast(ModelSamplingDiscrete, m.get_model_object("model_sampling"))
        m.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.CALC_COND_BATCH,
            WRAPPER_KEY,
            anima_ref_window_wrapper(
                state,
                sampling.percent_to_sigma(start_percent),
                sampling.percent_to_sigma(end_percent),
            ),
        )


NODES = [AnimaRefPPM]
