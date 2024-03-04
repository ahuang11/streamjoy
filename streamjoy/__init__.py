import logging

from ._utils import update_logger
from .core import stream, connect
from .models import GifStream, Mp4Stream
from .renderers import (
    default_holoviews_renderer,
    default_pandas_renderer,
    default_xarray_renderer,
)
from .settings import config, file_handlers, obj_handlers

__version__ = "0.0.0"

__all__ = [
    "config",
    "default_holoviews_renderer",
    "default_pandas_renderer",
    "default_xarray_renderer",
    "file_handlers",
    "obj_handlers",
    "stream",
    "connect",
    "GifStream",
    "Mp4Stream",
]


logging.basicConfig(
    level=config["logging_level"],
    format=config["logging_format"],
    datefmt=config["logging_datefmt"],
)

update_logger()