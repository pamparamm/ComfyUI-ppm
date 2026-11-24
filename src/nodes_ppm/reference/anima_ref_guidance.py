import math
from typing import Any, Callable, cast

import torch
import torch.nn.functional as F

import comfy.samplers
from comfy.model_patcher import ModelPatcher
from comfy.model_sampling import ModelSamplingDiscrete
from comfy_api.latest import io

from .anima_ref import REF_BIAS_KEY, REF_LATENTS_KEY, REF_TEXT_KEY, strip_ref_conds


def reference_values(conds: list) -> dict[str, Any] | None:
    for cond in conds:
        if REF_LATENTS_KEY in cond[1]:
            return {key: cond[1][key] for key in (REF_LATENTS_KEY, REF_TEXT_KEY, REF_BIAS_KEY) if key in cond[1]}
    return None


def carry_reference(conds: list, reference: dict[str, Any], offset: float) -> list:
    if offset != 0.0 and reference.get(REF_BIAS_KEY):
        shifted = tuple(bias + offset for bias in reference[REF_BIAS_KEY])
        reference = {**reference, REF_BIAS_KEY: shifted}
    return [[cond[0], {**cond[1], **reference}] for cond in conds]


def reference_correction(guided: torch.Tensor, cond: torch.Tensor, correction: str, multiplier: float) -> torch.Tensor:
    if correction == "apg":
        dims = tuple(range(1, guided.ndim))
        update = (guided - cond).float()
        unit = F.normalize(cond.float().flatten(1), dim=1).view_as(cond)
        update = update - (update * unit).sum(dim=dims, keepdim=True) * unit
        corrected = cond + update.to(guided.dtype)
    elif correction == "rescale":
        dims = tuple(range(1, guided.ndim))
        ratio = cond.std(dim=dims, keepdim=True) / guided.std(dim=dims, keepdim=True).clamp(min=1e-8)
        corrected = guided * ratio.to(guided.dtype)
    else:
        return guided
    return corrected * multiplier + guided * (1.0 - multiplier)


def anima_ref_guidance_post_cfg(
    ref_guidance: float, correction: str, multiplier: float, sigma_start: float, sigma_end: float
) -> Callable[[dict[str, Any]], torch.Tensor]:
    def _anima_ref_guidance_post_cfg(args: dict[str, Any]) -> torch.Tensor:
        sigma: float = float(args["sigma"][0])
        if sigma > sigma_start or sigma <= sigma_end:
            return args["denoised"]
        guided, uncond, uncond_denoised = args["denoised"], args["uncond"], args["uncond_denoised"]
        if ref_guidance != 1.0 and uncond is not None and uncond_denoised is not None:
            (base,) = comfy.samplers.calc_cond_batch(
                args["model"],
                [strip_ref_conds(uncond)],  # pyrefly: ignore[bad-argument-type]
                args["input"],
                args["sigma"],
                args["model_options"],
            )
            guided = guided + (ref_guidance - 1.0) * (uncond_denoised - base)
        return reference_correction(guided, args["cond_denoised"], correction, multiplier)

    return _anima_ref_guidance_post_cfg


class AnimaRefGuidancePPM(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AnimaRefGuidancePPM",
            display_name="Anima Reference Guidance (PPM)",
            category="model/patch/anima",
            description="Increases reference influence of Anima Reference (PPM) by comparing the sample with reference to the same sample without reference.\n"
            "Put it after Anima Reference (PPM) and before other post-CFG nodes.",
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Float.Input(
                    "ref_guidance",
                    default=1.0,
                    min=0.0,
                    max=20.0,
                    step=0.1,
                    tooltip="Controls how strongly to push away from a reference-free sample on top of the CFG scale.\n"
                    "Costs extra model call per step when set to a value other than 1.0.",
                ),
                io.Float.Input(
                    "fidelity",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip="Controls reference image injection into negative cond. 1.0 keeps it out of negative cond, and 0.0 fully injects it into negative cond.",
                ),
                io.Combo.Input(
                    "correction",
                    options=["none", "apg", "rescale"],
                    default="none",
                ),
                io.Float.Input(
                    "multiplier",
                    default=0.7,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Correction strength.",
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
        model: ModelPatcher = kwargs["model"]
        positive, negative = kwargs["positive"], kwargs["negative"]
        ref_guidance: float = float(kwargs["ref_guidance"])
        fidelity: float = float(kwargs["fidelity"])
        correction: str = kwargs["correction"]
        multiplier: float = float(kwargs["multiplier"])
        reference = reference_values(positive) or reference_values(negative)
        if reference is None or (ref_guidance == 1.0 and fidelity == 1.0 and correction == "none"):
            return io.NodeOutput(model, positive, negative)

        if ref_guidance != 1.0 or fidelity < 1.0:
            offset = 0.0 if fidelity >= 1.0 else math.log(max(1.0 - fidelity, 1e-6))
            negative = carry_reference(negative, reference, offset)

        if ref_guidance == 1.0 and correction == "none":
            return io.NodeOutput(model, positive, negative)

        m: ModelPatcher = model.clone()
        sampling = cast(ModelSamplingDiscrete, m.get_model_object("model_sampling"))
        m.set_model_sampler_post_cfg_function(
            anima_ref_guidance_post_cfg(
                ref_guidance,
                correction,
                multiplier,
                sampling.percent_to_sigma(kwargs["start_percent"]),
                sampling.percent_to_sigma(kwargs["end_percent"]),
            ),
            disable_cfg1_optimization=ref_guidance != 1.0,
        )
        return io.NodeOutput(m, positive, negative)


NODES = [AnimaRefGuidancePPM]
