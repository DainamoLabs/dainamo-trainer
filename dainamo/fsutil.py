"""File helpers shared by the server and the training processes."""

from __future__ import annotations

import os
import time


def replace(src: str, dst: str) -> None:
    """``os.replace``, retried for a moment: on Windows it fails while another process has ``dst`` open."""
    for attempt in range(40):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(0.05)
