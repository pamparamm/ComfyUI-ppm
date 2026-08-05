from typing import TYPE_CHECKING

# Absolute imports for pyrefly typechecking
if TYPE_CHECKING:
    from src.extension import comfy_entrypoint, WEB_DIRECTORY
else:
    from .src.extension import comfy_entrypoint as comfy_entrypoint, WEB_DIRECTORY as WEB_DIRECTORY
