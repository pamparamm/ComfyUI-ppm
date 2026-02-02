# ComfyUI-ppm

Just a bunch of some random nodes modified/fixed/created by me and/or others. If any node starts throwing errors after an update - try to delete and re-add the node.

You can browse example workflows for FLUX and SDXL NoobAI inside ComfyUI's "Browse Templates/Custom Nodes/ComfyUi-ppm" menu. I'll probably add some more examples in future (but I'm kinda lazy, kek).

## Nodes

### CLIPNegPip

> [!TIP]
>
> See "attention_couple+negpip" or "flux_negpip" from ComfyUI's "Browse Templates" menu.

Allows you to use negative weights in prompts. Negative weights can be used in both CFG models (such as SDXL) and Guidance-Distilled models (such as Flux) to negate a concept or a trait.

Supports:

- SD1
- SDXL
- Anima
- ~~FLUX~~ Unmaintained

Modified implementation of NegPiP by [laksjdjf](https://github.com/laksjdjf) and [hako-mikan](https://github.com/hako-mikan). It uses ModelPatcher instead of monkey-patching, which should increase compatibility with other nodes.

You can read more about NegPiP [in the original repo](https://github.com/hako-mikan/sd-webui-negpip). When used together with tag-based models, you should keep all commas inside weight braces (i.e. `(worst quality,:-1.3) (sketch,:-1.1)` instead of `(worst quality:-1.3), (sketch:-1.1),`).

> [!NOTE]
> `CLIPNegPip` is compatible with:
>
> - [ComfyUI prompt control by asagi4](https://github.com/asagi4/comfyui-prompt-control/)
> - [Advanced CLIP Text Encode extension by BlenderNeko](https://github.com/BlenderNeko/ComfyUI_ADV_CLIP_emb)
>
> `CLIPNegPip` is **incompatible** with:
>
> - [smZ Nodes by shiimizu](https://github.com/shiimizu/ComfyUI_smZNodes) (ComfyUI prompt control can replace most of its functionality)
> - [Comfyui_Flux_Style_Adjust by yichengup](https://github.com/yichengup/Comfyui_Flux_Style_Adjust) (and probably some other custom nodes that modify cond tensors)

### AttentionCouplePPM

> [!TIP]
>
> See "attention_couple+negpip" and "attention_couple_with_global_prompt" from ComfyUI's "Browse Templates" menu.

> [!NOTE]
> You can also use the version from [ComfyUI prompt control](https://github.com/asagi4/comfyui-prompt-control/) ([doc](https://github.com/asagi4/comfyui-prompt-control/blob/master/doc/attention_couple.md)) - it has a convenient prompt-based approach that doesn't require to add new `cond`/`mask` nodes for each new region.

Modified implementation of AttentionCouple by [laksjdjf](https://github.com/laksjdjf) and [Haoming02](https://github.com/Haoming02), made to be more compatible with other custom nodes.

Inputs for new regions are managed automatically: when you attach cond/mask of a region to the node, a new `cond_` / `mask_` input appears. Link `base_cond` input to the `positive` conditioning used in `KSampler`/`SamplerCustom`.

You can use multiple `LatentToMaskBB` nodes to set bounding boxes for `AttentionCouplePPM`. The parameters are relative to your initial latent: `x=0.5, y=0.0, w=0.5, h=1.0` will produce a mask covering the right half of the image.

You can adjust mask values to set region strength and use `ConditioningSetAreaStrength` to increase/decrease conditioning strength.

### Attention Selectors

`Model Attention Selector` and `CLIP Attention Selector` nodes, can be used to swap the optimized attention algorithm, such as pytorch (SDPA), sage (SageAttention), xformers, etc., without restarting ComfyUI.

### DynSamplerSelect

Modified samplers from [Euler-Smea-Dyn-Sampler by Koishi-Star](https://github.com/Koishi-Star/Euler-Smea-Dyn-Sampler).

Contains some new samplers: `euler_ancestral_dy`, `dpmpp_2m_dy` and `dpmpp_3m_dy`.

Tweaking `s_dy_pow` may reduce blur artifacts (optimal value is `2` for `euler_*` samplers and `-1` for `dpmpp_*` samplers, use `-1` to disable this feature).

### CFG++SamplerSelect

Samplers adapted to [CFG++: Manifold-constrained Classifier Free Guidance for Diffusion Models by Chung et al.](https://cfgpp-diffusion.github.io/). Also contains converted samplers from Euler-Smea-Dyn.

Should greatly reduce overexposure effect. Use together with `SamplerCustom` node. Don't forget to set CFG scale to 1.0-2.0 and PAG/SEG scale (if used) to 0.5-1.0.

### Guidance Limiter

Implementation of [Applying Guidance in a Limited Interval Improves Sample and Distribution Quality in Diffusion Models by Kynkäänniemi et al.](https://arxiv.org/abs/2404.07724) as a post CFG function.

Guidance Limiter is also available as a `CFGLimiterGuider` guider node for `SamplerCustomAdvanced`.

### Epsilon Scaling (PPM)

Modified version of ComfyUI's `Epsilon Scaling` node with a hacky (and mathematically incorrect) support for v-pred ZSNR models.

### Tile Preprocessor (PPM)

Image preprocessor for ControlNet Tile that doesn't require any third-party libraries (aside from `kornia`, which is a part of ComfyUI's requirements).

### Post-CFG nodes

Post-CFG variants of some nodes - they should have increased compatibility with other CFG-related nodes, making it possible to chain them together:

- `RescaleCFGPost`
- `RenormCFGPost`
- `DynamicThresholdingSimplePost` and `DynamicThresholdingFullPost` (based on [sd-dynamic-thresholding by mcmonkey4eva](https://github.com/mcmonkeyprojects/sd-dynamic-thresholding))

### Empty Latent Image (Aspect Ratio)

Generates empty latent with specified aspect ratio and with respect to target resolution.

### CLIPTextEncodeBREAK

A small lightweight wrapper over `ConditioningConcat` node. It splits prompts into chunks by `BREAK` keyword and produces a single concatenated conditioning.

### CLIPTokenCounter

Counts tokens in your prompt and returns them as a string. You can use `Preview Any` node to display them.

## Hooks/Hijacks

### Schedulers

Adds some schedulers to the default list from ComfyUI by replacing `comfy.samplers.calculate_sigmas` function:

- `ays` and `ays+` from [AlignYourSteps scheduler modified by Extraltodeus](https://github.com/Extraltodeus/sigmas_tools_and_the_golden_scheduler/blob/0dc89a264ef346a093d053c0da751f3ece317613/sigmas_merge.py#L203-L233) - `ays` is the default AYS scheduler (SDXL variant) and `ays+` is just `ays` with `force_sigma_min=True`
- `ays_30` and `gits` schedulers based on [AYS_32 by Koitenshin](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/15751#issuecomment-2143648234)
- `beta_1_1` - ComfyUI's beta scheduler with both alpha and beta set to 1.0
