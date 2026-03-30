from ._version import version as __version__
try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from napari_cellspots._widget1 import CellspotsProcessingWidget
from napari_cellspots._widget2 import CellspotsPolarWidget

__all__ = ["CellspotsProcessingWidget", "CellspotsPolarWidget"]
