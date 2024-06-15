# ComfyUI-ppm
Just a bunch of random nodes modified/fixed/created by me or others. If any node starts throwing errors after an update - try re-adding it.

I'll add some example workflows soon.

# Nodes

## CLIPNegPip
Modified implementation of NegPiP by [laksjdjf](https://github.com/laksjdjf) and [hako-mikan](https://github.com/hako-mikan). It uses ModelPatcher instead of monkey-patching, which should increase compatibility with other nodes.

`CLIPNegPip` node allows you to use negative weights in prompts. You should connect CLIPNegPip before other model/clip patches. After that, you can enter negative weights in your prompts (CTRL + arrows hotkey is capped at 0.0, will probably fix that soon).

You can read more about NegPiP [in the original repo](https://github.com/hako-mikan/sd-webui-negpip). I recommend putting everything from negative prompt to positive with a negative weight of something like -1.1 or -1.3. It's also better to keep all commas inside weight braces (i.e. `(worst quality,:-1.3) (sketch:-1.1,)` instead of `(worst quality:-1.3), (sketch:-1.1),`).

## AttentionCouplePPM
Modified implementation of AttentionCouple by [laksjdjf](https://github.com/laksjdjf) and [Haoming02](https://github.com/Haoming02). I made `AttentionCouplePPM` node compatible with `CLIPNegPiP` node and with default `PatchModelAddDownscale (Kohya Deep Shrink)` node. You can add/remove regions by right-clicking the node and selecting `Add Region`/`Remove Region`.

You can use multiple `LatentToMaskBB` nodes to set bounding box masks for `AttentionCouplePPM`. The parameters are relative to your initial latent: `x=0.5, y=0.0, w=0.5, h=1.0` will produce a mask covering right half of your image.

## Empty Latent Image (Aspect Ratio)
`Empty Latent Image (Aspect Ratio)` node generates empty latent with specified aspect ratio and with respect to target resolution.

## CLIPTextEncodeBREAK
A small lightweight wrapper over `ConditioningConcat` node, `CLIPTextEncodeBREAK` node can split prompts by `BREAK` keyword into chunks and produce a single concatenated conditioning.

## CLIPTokenCounter
Counts tokens in your prompt and returns them as a string. You can also print token count + individual tokens by enabling `debug_print`.

# Hooks/Hijacks

## Schedulers
Adds [AlignYourSteps scheduler modified by Extraltodeus](https://github.com/Extraltodeus/sigmas_tools_and_the_golden_scheduler/blob/0dc89a264ef346a093d053c0da751f3ece317613/sigmas_merge.py#L203-L233) to the default list of schedulers by replacing `comfy.samplers.calculate_sigmas` function. `ays` is the default AYS scheduler and `ays+` is just `ays` with `force_sigma_min=True`.