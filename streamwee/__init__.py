import logging

from .core import stream
from .settings import config
from .models import GifStream, Mp4Stream

__version__ = "0.0.0"

__all__ = ["config", "stream", "GifStream", "Mp4Stream"]


logging.basicConfig(
    level=config["logging_level"].upper(),
    format=config["logging_format"],
    datefmt=config["logging_datefmt"],
)