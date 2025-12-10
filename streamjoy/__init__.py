import logging

from ._utils import update_logger
from .core import connect, side_by_side, stream
from .models import ImageText, Paused
from .renderers import (
    default_holoviews_renderer,
    default_pandas_renderer,
    default_xarray_renderer,
)
from .settings import config, file_handlers, obj_handlers
from .streams import GifStream, HtmlStream, Mp4Stream, SideBySideStreams
from .wrappers import wrap_holoviews, wrap_matplotlib

__version__ = "0.0.10"

__all__ = [
    "GifStream",
    "HtmlStream",
    "ImageText",
    "Mp4Stream",
    "Paused",
    "SideBySideStreams",
    "config",
    "connect",
    "default_holoviews_renderer",
    "default_pandas_renderer",
    "default_xarray_renderer",
    "file_handlers",
    "obj_handlers",
    "side_by_side",
    "stream",
    "wrap_holoviews",
    "wrap_matplotlib",
]

logging.basicConfig(
    level=config["logging_level"],
    format=config["logging_format"],
    datefmt=config["logging_datefmt"],
)

update_logger()
