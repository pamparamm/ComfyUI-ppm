from typing import TYPE_CHECKING

# Absolute imports for pyrefly typechecking
if TYPE_CHECKING:
    from src.extension import WEB_DIRECTORY, comfy_entrypoint
else:
    from .src.extension import WEB_DIRECTORY as WEB_DIRECTORY
    from .src.extension import comfy_entrypoint as comfy_entrypoint
