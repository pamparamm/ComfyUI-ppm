import torch

from comfy.comfy_types.node_typing import IO, ComfyNodeABC, InputTypeDict
from comfy.sd import VAE, AutoencoderKL


class VAEPadding(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "vae": (IO.VAE, {}),
                "mode": (
                    IO.COMBO,
                    {
                        "default": "reflect",
                        "options": ["zeros", "reflect"],
                    },
                ),
            }
        }

    RETURN_TYPES = (IO.VAE,)
    FUNCTION = "patch"

    CATEGORY = "latent"
    DESCRIPTION = "Sets padding mode for all Conv2D layers in VAE (necessary for some VAEs like `Anzhc/MS-LC-EQ-D-VR_VAE`)"

    def patch(self, vae: VAE, mode: str):
        if not isinstance(vae.first_stage_model, AutoencoderKL):
            return (vae,)

        for module in vae.first_stage_model.modules():
            if isinstance(module, torch.nn.Conv2d):
                pad_h, pad_w = (
                    module.padding if isinstance(module.padding, tuple) else (int(module.padding), int(module.padding))
                )
                if pad_h > 0 or pad_w > 0:
                    module.padding_mode = mode

        return (vae,)


NODE_CLASS_MAPPINGS = {
    "VAEPadding": VAEPadding,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VAEPadding": "Set VAE Padding",
}
