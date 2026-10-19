import numpy as np
import torch
from PIL import Image

import nodes
from comfy_api.latest import io
from comfy_extras.nodes_bounding_boxes import (
    boxes_from_input,
    boxes_to_regions,
    build_elements,
    fractions_to_bbox_frame,
    render_preview,
)


def _to_pil(image: torch.Tensor) -> Image.Image:
    arr = (image[0].detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _canvas_in_frame(boxes: list, width: int, height: int) -> list:
    """Put canvas boxes in the frame the node is running with.

    The canvas draws in relative coordinates and stamps the pixel frame it wrote a
    box in, so a frame change (linked width/height) keeps the boxes where they are
    instead of reading the old pixels against the new frame.
    """
    framed = []
    for box in boxes:
        meta = box.get("metadata") if isinstance(box, dict) else None
        frame = meta.get("frame") if isinstance(meta, dict) else None
        if not (
            isinstance(frame, (list, tuple))
            and len(frame) == 2
            and all(isinstance(v, (int, float)) and v > 0 for v in frame)
        ):
            framed.append(box)
            continue
        sx, sy = width / frame[0], height / frame[1]
        framed.append({
            **box,
            "x": box.get("x", 0) * sx,
            "y": box.get("y", 0) * sy,
            "width": box.get("width", 0) * sx,
            "height": box.get("height", 0) * sy,
        })
    return framed


class CreateBoundingBoxesPPM(io.ComfyNode):
    """Create Bounding Boxes with a canvas editor that also works on the classic canvas."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CreateBoundingBoxesPPM",
            display_name="Create Bounding Boxes (PPM)",
            category="utilities",
            description=(
                "Draw bounding boxes in a canvas. Outputs prompt elements, pixel-space bounding boxes and a preview image. "
                "The canvas editor is available both in Nodes 2.0 and with Nodes 2.0 disabled."
            ),
            inputs=[
                io.Image.Input(
                    "background",
                    optional=True,
                    tooltip="Optional image used as background in the canvas and preview.",
                ),
                io.MultiType.Input(
                    "bboxes",
                    [io.BoundingBox, io.Array, io.String],
                    optional=True,
                    tooltip=(
                        "Bounding boxes, elements, or a JSON string to initialize the canvas. A new upstream value "
                        "initializes the canvas; edits made on the canvas take priority and are kept until the upstream "
                        "value changes again."
                    ),
                ),
                io.Int.Input(
                    "width",
                    default=1024,
                    min=64,
                    max=16384,
                    step=16,
                    tooltip="Width of the canvas and the pixel grid for the bounding boxes.",
                ),
                io.Int.Input(
                    "height",
                    default=1024,
                    min=64,
                    max=16384,
                    step=16,
                    tooltip="Height of the canvas and the pixel grid for the bounding boxes.",
                ),
                io.BoundingBoxes.Input(
                    "editor_state",
                    socketless=False,
                    tooltip=(
                        "Draw bounding boxes and set each box type, text, description, color palette. "
                        "Start with background element first and foreground last."
                    ),
                ),
                io.BoundingBoxes.Input(
                    "last_incoming",
                    optional=True,
                    tooltip=(
                        "Internal state managed by the canvas: the upstream bboxes value that last initialized it. "
                        "Leave empty to re-initialize the canvas from the bboxes input on the next run."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(display_name="preview"),
                io.BoundingBox.Output(display_name="bboxes"),
                io.Array.Output(display_name="elements"),
            ],
            is_experimental=True,
            is_output_node=True,
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        width: int = kwargs["width"]
        height: int = kwargs["height"]
        background = kwargs.get("background")
        bboxes = kwargs.get("bboxes")
        editor_state = kwargs.get("editor_state")
        last_incoming = kwargs.get("last_incoming")

        incoming = boxes_from_input(bboxes, width, height)
        applied = last_incoming if isinstance(last_incoming, list) else []
        upstream_changed = bool(incoming) and incoming != applied
        source = incoming if upstream_changed else _canvas_in_frame(editor_state or [], width, height)
        regions = boxes_to_regions(source, width, height)

        has_background = background is not None and len(background) > 0
        preview = render_preview(regions, width, height, _to_pil(background) if has_background else None)

        ui = {"dims": [width, height]}
        if incoming:
            ui["input_bboxes"] = incoming
        if has_background:
            saved = nodes.PreviewImage().save_images(background[:1], "comfy.bboxes.background")
            ui["background_images"] = saved["ui"]["images"]

        return io.NodeOutput(
            preview,
            fractions_to_bbox_frame(regions, width, height),
            build_elements(regions),
            ui=ui,
        )


NODES = [CreateBoundingBoxesPPM]
