import logging

from ._utils import update_logger
from .core import stream, connect
from .streams import GifStream, Mp4Stream
from .models import Paused, ImageText
from .renderers import (
    default_holoviews_renderer,
    default_pandas_renderer,
    default_xarray_renderer,
)
from .wrappers import wrap_holoviews, wrap_matplotlib
from .settings import config, file_handlers, obj_handlers

__version__ = "0.0.0"
__all__ = [
    "GifStream",
    "ImageText",
    "Mp4Stream",
    "Paused",
    "config",
    "connect",
    "default_holoviews_renderer",
    "default_pandas_renderer",
    "default_xarray_renderer",
    "file_handlers",
    "obj_handlers",
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
