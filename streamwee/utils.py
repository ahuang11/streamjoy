from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from dask.distributed import Client, Future, get_client

from .settings import config

logger = logging.getLogger(__name__)


def update_logger(
    level: str | None = None,
    format: str | None = None,
    datefmt: str | None = None,
) -> None:
    level = level or config["logging_level"]
    format = format or config["logging_format"]
    datefmt = datefmt or config["logging_datefmt"]
    for handler in logging.getLogger().handlers:
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(format, datefmt))


def warn_default_used(
    key: str, default_value: Any, total_value: Any | None = None, suffix: str = ""
) -> None:
    color = config["logging_color"]
    reset = config["logging_reset"]
    message = (
        f"No {color}{key!r}{reset} specified; using the default: "
        f"{color}{default_value!r}{reset}"
    )
    if total_value:
        message += f" / {total_value!r}"
    if suffix:
        message += f" {suffix}"
    message += f". To suppress this warning, pass {key!r}."
    logging.warn(message)


def get_config_default(key: str, value: Any, warn: bool = True) -> Any:
    if value is None:
        default = config[key]
        if warn:
            value = warn_default_used(key, default)
        value = default
    return value


def get_distributed_client(client: Client | None = None, **kwargs) -> Client:
    if client is not None:
        return client

    try:
        client = get_client()
    except ValueError:
        client = Client(**kwargs)
    return client


def download_file(base_url: str, scratch_dir: Path, file: str) -> Path:
    import httpx

    url = base_url + file
    path = scratch_dir / file
    if path.exists():
        return path

    with httpx.stream("GET", url) as response:
        response.raise_for_status()
        with open(path, "wb") as buf:
            for chunk in response.iter_bytes():
                buf.write(chunk)
    return path


def get_max_frames(total_frames: int, max_frames: int) -> int:
    default_max_frames = config["max_frames"]
    if max_frames is None and total_frames > default_max_frames:
        warn_default_used(
            "max_frames", default_max_frames, total_value=total_frames, suffix="frames."
        )
        max_frames = default_max_frames
    elif max_frames is None:
        max_frames = total_frames
    return max_frames


def get_result(future: Future) -> Any:
    if isinstance(future, Future):
        return future.result()
    elif hasattr(future, "compute"):
        return future.compute()
    else:
        return future


def using_notebook():
    try:
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:  # Check if under IPython kernel
            return False
    except Exception:
        return False
    return True
