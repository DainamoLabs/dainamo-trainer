"""Registry of model families."""

from .base import FamilyDriver
from .krea2.driver import Krea2Driver
from .ltx2.driver import LTX25Driver
from .minimax_h3.driver import MinimaxH3Driver
from .moss_image.driver import MossImageDriver
from .qwen_image_2.driver import QwenImage2Driver
from .yue2.driver import YuE2Driver

FAMILIES: dict[str, type[FamilyDriver]] = {
    Krea2Driver.info.id: Krea2Driver,
    MossImageDriver.info.id: MossImageDriver,
    MinimaxH3Driver.info.id: MinimaxH3Driver,
    QwenImage2Driver.info.id: QwenImage2Driver,
    LTX25Driver.info.id: LTX25Driver,
    YuE2Driver.info.id: YuE2Driver,
}


def get_family(family_id: str) -> type[FamilyDriver]:
    if family_id not in FAMILIES:
        raise KeyError(f"Unknown model family {family_id!r}")
    return FAMILIES[family_id]
