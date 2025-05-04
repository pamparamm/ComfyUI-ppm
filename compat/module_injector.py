import logging
import os
import sys
from types import ModuleType
from typing import Callable

import folder_paths
from nodes import get_module_name


class ModuleInjector:
    """Injector for monkey-patching"""

    def __init__(self, name: str, modules: list[ModuleType]) -> None:
        self.name = name
        self.modules = modules

    def patch(self, patch_func: Callable[[ModuleType], None]):
        logging.info(f"{self.name} was patched by ComfyUI-ppm")
        for module in self.modules:
            patch_func(module)


# Based on init_external_custom_nodes method from nodes.py (ComfyUI)
def get_module_injector(module_name: str):
    modules: list[ModuleType] = []
    node_paths = folder_paths.get_folder_paths("custom_nodes")
    for custom_node_path in node_paths:
        possible_modules = os.listdir(os.path.realpath(custom_node_path))
        if "__pycache__" in possible_modules:
            possible_modules.remove("__pycache__")

        for possible_module in [m for m in possible_modules if module_name.lower() in m.lower()]:
            module_path = os.path.join(custom_node_path, possible_module)
            if os.path.isfile(module_path) and os.path.splitext(module_path)[1] != ".py":
                continue
            if module_path.endswith(".disabled"):
                continue
            module_candidate = get_custom_node(module_path)
            if module_candidate is not None:
                modules.append(module_candidate)
    return ModuleInjector(module_name, modules)


# Based on load_custom_node method from nodes.py (ComfyUI)
def get_custom_node(module_path):
    module_name = get_module_name(module_path)
    if os.path.isfile(module_path):
        sp = os.path.splitext(module_path)
        module_name = sp[0]
        sys_module_name = module_name
    elif os.path.isdir(module_path):
        sys_module_name = module_path.replace(".", "_x_")
    else:
        return None

    if sys_module_name in sys.modules:
        return sys.modules[sys_module_name]

    return None
