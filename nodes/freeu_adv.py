# modified (and partially simplified) version of https://github.com/WASasquatch/FreeU_Advanced (MIT License)
# code originally taken from: https://github.com/ChenyangSi/FreeU (under MIT License)

from functools import partial
import torch
import torch as th
import torch.fft as fft
from comfy.ldm.modules.diffusionmodules.openaimodel import forward_timestep_embed, apply_control
from comfy.ldm.modules.diffusionmodules.util import timestep_embedding


def _forward(_self, x, timesteps=None, context=None, y=None, control=None, transformer_options={}, **kwargs):
    """
    Apply the model to an input batch.
    :param x: an [N x C x ...] Tensor of inputs.
    :param timesteps: a 1-D batch of timesteps.
    :param context: conditioning plugged in via crossattn
    :param y: an [N] Tensor of labels, if class-conditional.
    :return: an [N x C x ...] Tensor of outputs.
    """
    transformer_options["original_shape"] = list(x.shape)
    transformer_options["transformer_index"] = 0
    transformer_patches = transformer_options.get("patches", {})

    num_video_frames = kwargs.get("num_video_frames", _self.default_num_video_frames)
    image_only_indicator = kwargs.get("image_only_indicator", getattr(_self, "default_image_only_indicator", None))
    time_context = kwargs.get("time_context", None)

    assert (y is not None) == (_self.num_classes is not None), "must specify y if and only if the model is class-conditional"
    hs = []
    t_emb = timestep_embedding(timesteps, _self.model_channels, repeat_only=False).to(x.dtype)
    emb = _self.time_embed(t_emb)

    if "emb_patch" in transformer_patches:
        patch = transformer_patches["emb_patch"]
        for p in patch:
            emb = p(emb, _self.model_channels, transformer_options)

    if _self.num_classes is not None:
        assert y.shape[0] == x.shape[0]
        emb = emb + _self.label_emb(y)

    h = x
    for id, module in enumerate(_self.input_blocks):
        transformer_options["block"] = ("input", id)
        h = forward_timestep_embed(
            module,
            h,
            emb,
            context,
            transformer_options,
            time_context=time_context,
            num_video_frames=num_video_frames,
            image_only_indicator=image_only_indicator,
        )
        h = apply_control(h, control, "input")
        if "input_block_patch" in transformer_patches:
            patch = transformer_patches["input_block_patch"]
            for p in patch:
                h = p(h, transformer_options)

        hs.append(h)
        if "input_block_patch_after_skip" in transformer_patches:
            patch = transformer_patches["input_block_patch_after_skip"]
            for p in patch:
                h = p(h, transformer_options)

    transformer_options["block"] = ("middle", 0)
    h = forward_timestep_embed(
        _self.middle_block,
        h,
        emb,
        context,
        transformer_options,
        time_context=time_context,
        num_video_frames=num_video_frames,
        image_only_indicator=image_only_indicator,
    )
    h = apply_control(h, control, "middle")

    if "middle_block_patch" in transformer_patches:
        patch = transformer_patches["middle_block_patch"]
        for p in patch:
            h = p(h, transformer_options)

    for id, module in enumerate(_self.output_blocks):
        transformer_options["block"] = ("output", id)
        hsp = hs.pop()
        hsp = apply_control(hsp, control, "output")

        if "output_block_patch" in transformer_patches:
            patch = transformer_patches["output_block_patch"]
            for p in patch:
                h, hsp = p(h, hsp, transformer_options)

        h = th.cat([h, hsp], dim=1)
        del hsp
        if len(hs) > 0:
            output_shape = hs[-1].shape
        else:
            output_shape = None
        h = forward_timestep_embed(
            module,
            h,
            emb,
            context,
            transformer_options,
            output_shape,
            time_context=time_context,
            num_video_frames=num_video_frames,
            image_only_indicator=image_only_indicator,
        )
    h = h.type(x.dtype)
    if _self.predict_codebook_ids:
        return _self.id_predictor(h)
    else:
        return _self.out(h)


def Fourier_filter(x, threshold, scale):
    # FFT
    if isinstance(x, list):
        x = x[0]
    if isinstance(x, torch.Tensor):
        x_freq = fft.fftn(x.float(), dim=(-2, -1))
        x_freq = fft.fftshift(x_freq, dim=(-2, -1))

        B, C, H, W = x_freq.shape
        mask = torch.ones((B, C, H, W), device=x.device)

        crow, ccol = H // 2, W // 2
        mask[..., crow - threshold : crow + threshold, ccol - threshold : ccol + threshold] = scale

        x_freq = x_freq * mask

        # IFFT
        x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
        x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real

        return x_filtered.to(x.dtype)

    return x


class FreeU2PPM:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "input_block": ("BOOLEAN", {"default": False}),
                "middle_block": ("BOOLEAN", {"default": False}),
                "output_block": ("BOOLEAN", {"default": False}),
                "slice_b1": ("INT", {"default": 640, "min": 64, "max": 1280, "step": 1}),
                "slice_b2": ("INT", {"default": 320, "min": 64, "max": 640, "step": 1}),
                "b1": ("FLOAT", {"default": 1.1, "min": 0.0, "max": 10.0, "step": 0.001}),
                "b2": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 10.0, "step": 0.001}),
                "s1": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 10.0, "step": 0.001}),
                "s2": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 10.0, "step": 0.001}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": False}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": False}),
            },
            "optional": {
                "threshold": ("INT", {"default": 1.0, "max": 10, "min": 1, "step": 1}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "_for_testing"

    def patch(
        self,
        model,
        input_block,
        middle_block,
        output_block,
        slice_b1,
        slice_b2,
        b1,
        b2,
        s1,
        s2,
        threshold=1.0,
        start_percent=0.0,
        end_percent=1.0,
    ):
        model_sampling = model.get_model_object("model_sampling")
        sigma_start = model_sampling.percent_to_sigma(start_percent)
        sigma_end = model_sampling.percent_to_sigma(end_percent)

        min_slice = 64
        max_slice_b1 = 1280
        max_slice_b2 = 640
        slice_b1 = max(min(max_slice_b1, slice_b1), min_slice)
        slice_b2 = max(min(min(slice_b1, max_slice_b2), slice_b2), min_slice)

        def _hidden_mean(h):
            hidden_mean = h.mean(1).unsqueeze(1)
            B = hidden_mean.shape[0]
            hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True)
            hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
            hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)
            return hidden_mean

        def block_patch(h, transformer_options):
            sigma = transformer_options["sigmas"]
            if not (sigma_end < sigma[0] <= sigma_start):
                return h

            if h.shape[1] == 1280:
                hidden_mean = _hidden_mean(h)
                h[:, :slice_b1] = h[:, :slice_b1] * ((b1 - 1) * hidden_mean + 1)
            if h.shape[1] == 640:
                hidden_mean = _hidden_mean(h)
                h[:, :slice_b2] = h[:, :slice_b2] * ((b2 - 1) * hidden_mean + 1)
            return h

        def block_patch_hsp(h, hsp, transformer_options):
            sigma = transformer_options["sigmas"]
            if not (sigma_end < sigma[0] <= sigma_start):
                return h, hsp

            if h.shape[1] == 1280:
                h = block_patch(h, transformer_options)
                hsp = Fourier_filter(hsp, threshold=threshold, scale=s1)
            if h.shape[1] == 640:
                h = block_patch(h, transformer_options)
                hsp = Fourier_filter(hsp, threshold=threshold, scale=s2)
            return h, hsp

        m = model.clone()
        m.add_object_patch("diffusion_model.forward", partial(_forward, m.model.diffusion_model))
        if output_block:
            print("Patching output block")
            m.set_model_output_block_patch(block_patch_hsp)
        if input_block:
            print("Patching input block")
            m.set_model_input_block_patch(block_patch)
        if middle_block:
            print("Patching middle block")
            m.set_model_patch(block_patch, "middle_block_patch")
        return (m,)
