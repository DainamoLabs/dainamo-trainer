"""Registry of model families.

The drivers are imported on first use (``FAMILIES``, ``get_family``, ``load``), not with the package: they bring in
diffusers and each family's model code, about 2 s, which the server would otherwise spend before it can answer."""

from __future__ import annotations

import importlib
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import FamilyDriver

# (module, class), in the order the UI lists them
_DRIVERS = [
    (".krea2.driver", "Krea2Driver"),
    (".flux2_klein.driver", "Flux2KleinDriver"),
    (".flux3_klein.driver", "Flux3KleinDriver"),
    (".moss_image.driver", "MossImageDriver"),
    (".minimax_h3.driver", "MinimaxH3Driver"),
    (".qwen_image_2.driver", "QwenImage2Driver"),
    (".ltx2.driver", "LTX25Driver"),
    (".yue2.driver", "YuE2Driver"),
]
_lock = threading.Lock()
_families: dict[str, type[FamilyDriver]] | None = None


def load() -> dict[str, type[FamilyDriver]]:
    """{family id: driver class}, importing the drivers the first time."""
    global _families
    with _lock:
        if _families is None:
            classes = [getattr(importlib.import_module(m, __name__), c) for m, c in _DRIVERS]
            _families = {F.info.id: F for F in classes}
    return _families


def __getattr__(name: str):
    if name == "FAMILIES":
        return load()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_family(family_id: str) -> type[FamilyDriver]:
    families = load()
    if family_id not in families:
        raise KeyError(f"Unknown model family {family_id!r}")
    return families[family_id]
