from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from .compat.utils import v3_schema_stub
from .nodes_ppm import (
    attention_couple_ppm,
    attention_selector,
    clip_misc,
    clip_negpip,
    freeu_adv,
    guidance,
    image_misc,
    latent_misc,
    latent_tonemap,
    misc,
    samplers,
)
from .schedulers import inject_schedulers

WEB_DIRECTORY = "./js"


class PPMExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        # TODO convert more nodes to v3
        return [
            *attention_couple_ppm.NODES,
            *attention_selector.NODES,
            *v3_schema_stub(clip_misc),
            *clip_negpip.NODES,
            *v3_schema_stub(freeu_adv),
            *v3_schema_stub(guidance),
            *image_misc.NODES,
            *v3_schema_stub(latent_misc),
            *v3_schema_stub(latent_tonemap),
            *misc.NODES,
            *v3_schema_stub(samplers),
        ]

    inject_schedulers()
    attention_selector.init()


async def comfy_entrypoint():  # ComfyUI calls this to load your extension and its nodes.
    return PPMExtension()
