from typing import Callable

from comfy.ldm.modules import attention as attn
from comfy.model_patcher import ModelPatcher
from comfy.sd import CLIP
from comfy_api.latest import io

ATTENTIONS = ["optimized"]


class ModelAttentionSelector(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ModelAttentionSelector",
            display_name="Model Attention Selector",
            category="advanced/model",
            description="Replaces diffusion model's attention with another registered attention function",
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input(
                    "attention",
                    options=ATTENTIONS,
                    default=ATTENTIONS[0],
                ),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        model: ModelPatcher = kwargs["model"]
        attention: str = kwargs["attention"]

        m = model.clone()

        attention_function: Callable = attn.get_attention_function(attention)  # type: ignore

        def attention_override(_, *args, **kwargs):
            return attention_function(*args, **kwargs)

        options = m.model_options["transformer_options"].copy()
        options["optimized_attention_override"] = attention_override
        m.model_options["transformer_options"] = options

        return io.NodeOutput(m)


class CLIPAttentionSelector(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CLIPAttentionSelector",
            display_name="CLIP Attention Selector",
            category="advanced/model",
            description="Replaces text model's attention with another registered attention function",
            inputs=[
                io.Clip.Input("clip"),
                io.Combo.Input(
                    "attention",
                    options=ATTENTIONS,
                    default=ATTENTIONS[0],
                ),
            ],
            outputs=[
                io.Clip.Output(),
            ],
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        clip: CLIP = kwargs["clip"]
        attention: str = kwargs["attention"]

        c = clip.clone()
        patcher = c.patcher

        attention_function: Callable = attn.get_attention_function(attention)  # type: ignore

        def attention_override(_, *args, **kwargs):
            return attention_function(*args, **kwargs)

        options = patcher.model_options["transformer_options"].copy()
        options["optimized_attention_override"] = attention_override
        patcher.model_options["transformer_options"] = options

        return io.NodeOutput(c)


def init():
    ATTENTIONS.extend(attn.REGISTERED_ATTENTION_FUNCTIONS)


NODES = [ModelAttentionSelector, CLIPAttentionSelector]
