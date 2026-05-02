import torch
import torch.nn.functional as F

from comfy.model_management import get_torch_device
from comfy_api.latest import io

HAS_DEPS = True
try:
    from kornia.filters import filter2d
except ImportError:
    HAS_DEPS = False


class TilePreprocessorPPM(io.ComfyNode):
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic"]  # TODO lanczos/bislerp

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TilePreprocessorPPM",
            display_name="Tile Preprocessor (PPM)",
            category="image/preprocessors",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("blur_iters", default=3, min=1, max=10),
                io.Combo.Input(
                    "downscale_method",
                    options=cls.upscale_methods,
                    default="area",
                ),
                io.Combo.Input(
                    "upscale_method",
                    options=cls.upscale_methods,
                    default="bilinear",
                ),
                io.Boolean.Input("rescale_to_input", default=True),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        image: torch.Tensor = kwargs["image"]
        iters: int = kwargs["blur_iters"]
        downscale_method = kwargs["downscale_method"]
        upscale_method = kwargs["upscale_method"]
        rescale_to_input = kwargs["rescale_to_input"]

        image = image.to(get_torch_device())
        samples = image.movedim(-1, 1)

        h_orig, w_orig = samples.shape[2:4]
        downscale_factor = 2**iters

        samples_down = F.interpolate(
            samples,
            scale_factor=1 / downscale_factor,
            mode=downscale_method,
        )

        samples_up = samples_down
        for _ in range(iters):
            samples_up = cls.pyrup(samples_up, upscale_method)

        samples_result = (
            F.interpolate(
                samples_up,
                size=(h_orig, w_orig),
                mode="nearest-exact",
            )
            if rescale_to_input
            else samples_up
        )
        samples_result = samples_up.movedim(1, -1)

        return io.NodeOutput(samples_result)

    @classmethod
    def pyrup(cls, input: torch.Tensor, upscale_method="nearest-exact"):
        kernel = (
            torch.Tensor([
                [
                    [1.0, 4.0, 6.0, 4.0, 1.0],
                    [4.0, 16.0, 24.0, 16.0, 4.0],
                    [6.0, 24.0, 36.0, 24.0, 6.0],
                    [4.0, 16.0, 24.0, 16.0, 4.0],
                    [1.0, 4.0, 6.0, 4.0, 1.0],
                ]
            ])
            / 256.0
        )
        x_up = F.interpolate(input, scale_factor=2, mode=upscale_method)
        x_blur = filter2d(x_up, kernel, border_type="reflect")  # pyright: ignore[reportPossiblyUnboundVariable]
        return x_blur


NODES = []
if HAS_DEPS:
    NODES += [TilePreprocessorPPM]
