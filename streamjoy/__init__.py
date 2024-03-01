import logging

from .core import stream
from .models import GifStream, Mp4Stream
from .settings import config, readers

__version__ = "0.0.0"

__all__ = ["config", "readers", "stream", "patches", "GifStream", "Mp4Stream"]


logging.basicConfig(
    level=config["logging_level"],
    format=config["logging_format"],
    datefmt=config["logging_datefmt"],
)
