# src/methods/kds/stadion/__init__.py
from __future__ import annotations

from .kds import kds_loss
from ._version import __version__

__all__ = ["kds_loss", "__version__"]
