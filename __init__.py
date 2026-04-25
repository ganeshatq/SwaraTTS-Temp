"""Top-level package exports and legacy module aliases for ``swara_tts``."""

from importlib import import_module
import sys

from .models.core import *  # noqa: F401,F403
from .models.core import __all__ as _core_all
from .models.core.swara_model import SwaraTTS2Config

_LEGACY_MODULES = (
    "activations",
    "base",
    "convnet",
    "factory",
    "flow_matching",
    "istftnet",
    "norms",
    "resnet",
    "sampling",
    "source",
    "swara_model",
)

for _module_name in _LEGACY_MODULES:
    sys.modules[f"{__name__}.{_module_name}"] = import_module(
        f".models.core.{_module_name}",
        __name__,
    )

__all__ = [*_core_all, "SwaraTTS2Config"]
