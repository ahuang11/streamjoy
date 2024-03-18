import logging

from ._utils import update_logger
from .core import connect, stream
from .models import ImageText, Paused
from .renderers import (
    default_holoviews_renderer,
    default_pandas_renderer,
    default_xarray_renderer,
)
from .settings import config, file_handlers, obj_handlers
from .streams import GifStream, Mp4Stream
from .wrappers import wrap_holoviews, wrap_matplotlib

__version__ = "0.0.1"

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
