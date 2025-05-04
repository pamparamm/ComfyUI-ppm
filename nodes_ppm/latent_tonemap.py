import torch
from comfy.comfy_types.node_typing import IO, ComfyNodeABC, InputTypeDict


class LatentOperationTonemapLuminance(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "tonemapper": (IO.COMBO, {"default": "mobius", "options": ["reinhard", "mobius", "aces"]}),
                "multiplier": (IO.FLOAT, {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT_OPERATION",)  # type: ignore
    FUNCTION = "op"

    CATEGORY = "latent/advanced/operations"
    EXPERIMENTAL = True

    def op(self, tonemapper: str, multiplier: float):
        def tonemap_reinhard_luminance(latent, **kwargs):
            lum = latent[:, 0:1]

            lum_m = (torch.linalg.vector_norm(lum, dim=(1)) + 1e-10)[:, None]
            lum_norm = lum / lum_m

            mean = torch.mean(lum_m, dim=(1, 2, 3), keepdim=True)
            std = torch.std(lum_m, dim=(1, 2, 3), keepdim=True)
            top = (std * 5 + mean) * multiplier

            lum_m_t = lum_m / top
            lum_tonemap = 0.0

            if tonemapper == "reinhard":
                lum_tonemap = lum_m_t / (lum_m_t + 1.0)

            elif tonemapper == "mobius":
                lum_tonemap = (lum_m_t * (1 + lum_m_t)) / (1 + (lum_m_t * lum_m_t))

            elif tonemapper == "aces":
                lum_tonemap = (lum_m_t * (lum_m_t + 0.45)) / ((lum_m_t * lum_m_t) + 0.91 * lum_m_t + 0.91)

            tonemapped_lum = lum_norm * lum_tonemap * top

            output_latent = latent.clone()
            output_latent[:, 0:1] = tonemapped_lum
            return output_latent

        return (tonemap_reinhard_luminance,)
