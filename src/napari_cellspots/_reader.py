from pathlib import Path


def napari_get_reader(path):
    """Return a reader function for ICS/IDS files, or None if not applicable."""
    if isinstance(path, list):
        path = path[0]
    if Path(path).suffix.lower() not in ('.ics', '.ids'):
        return None
    return _reader_function


def _reader_function(path):
    """Read an ICS/IDS file using pyics and returns napari layer data."""
    import pyics

    if isinstance(path, list):
        path = path[0]
    path = Path(path)

    image_data, _meta = pyics.imread(path.as_posix())
    # image_data: (C, Z, H, W) — expose with channel_axis so napari splits channels
    return [(image_data, {"name": path.stem, "channel_axis": 0}, "image")]
