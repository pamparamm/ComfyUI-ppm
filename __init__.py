from .nodes_ppm import (
    attention_couple_ppm,
    clip_misc,
    clip_negpip,
    freeu_adv,
    guidance,
    latent_misc,
    latent_tonemap,
    misc,
    samplers,
    vae,
    attention_selector,
)
from .schedulers import hijack_schedulers

WEB_DIRECTORY = "./js"

NODE_CLASS_MAPPINGS = {
    **attention_couple_ppm.NODE_CLASS_MAPPINGS,
    **clip_misc.NODE_CLASS_MAPPINGS,
    **clip_negpip.NODE_CLASS_MAPPINGS,
    **freeu_adv.NODE_CLASS_MAPPINGS,
    **guidance.NODE_CLASS_MAPPINGS,
    **latent_misc.NODE_CLASS_MAPPINGS,
    **latent_tonemap.NODE_CLASS_MAPPINGS,
    **misc.NODE_CLASS_MAPPINGS,
    **samplers.NODE_CLASS_MAPPINGS,
    **vae.NODE_CLASS_MAPPINGS,
    **attention_selector.NODE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **attention_couple_ppm.NODE_DISPLAY_NAME_MAPPINGS,
    **clip_misc.NODE_DISPLAY_NAME_MAPPINGS,
    **clip_negpip.NODE_DISPLAY_NAME_MAPPINGS,
    **freeu_adv.NODE_DISPLAY_NAME_MAPPINGS,
    **guidance.NODE_DISPLAY_NAME_MAPPINGS,
    **latent_misc.NODE_DISPLAY_NAME_MAPPINGS,
    **latent_tonemap.NODE_DISPLAY_NAME_MAPPINGS,
    **misc.NODE_DISPLAY_NAME_MAPPINGS,
    **samplers.NODE_DISPLAY_NAME_MAPPINGS,
    **vae.NODE_DISPLAY_NAME_MAPPINGS,
    **attention_selector.NODE_DISPLAY_NAME_MAPPINGS,
}


hijack_schedulers()
