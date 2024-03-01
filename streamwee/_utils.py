from __future__ import annotations

import inspect
import logging
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

from dask.distributed import Client, Future, get_client

from .settings import config, readers


def update_logger(
    level: str | None = None,
    format: str | None = None,
    datefmt: str | None = None,
) -> logging.Logger:
    success_level = config["logging_level"]

    class CustomLogger(logging.Logger):
        def success(self, message, *args, **kws):
            if self.isEnabledFor(success_level):
                self._log(success_level, message, args, **kws)

    logging.setLoggerClass(CustomLogger)
    logging.addLevelName(success_level, "SUCCESS")
    logger = logging.getLogger(__name__)

    level = level or config["logging_level"]
    format = format or config["logging_format"]
    datefmt = datefmt or config["logging_datefmt"]
    for handler in logging.getLogger().handlers:
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(format, datefmt))

    return logger


def warn_default_used(
    key: str, default_value: Any, total_value: Any | None = None, suffix: str = ""
) -> None:
    color = config["logging_warning_color"]
    reset = config["logging_reset_color"]
    message = (
        f"No {color}{key!r}{reset} specified; using the default: "
        f"{color}{default_value!r}{reset}"
    )
    if total_value is not None:
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
            "max_frames",
            default_max_frames,
            total_value=total_frames,
            suffix="frames. Pass `-1` to use all frames",
        )
        max_frames = default_max_frames
    elif max_frames is None:
        max_frames = total_frames
    elif max_frames == -1:
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


def resolve_uri(in_memory: bool, scratch_dir: str | Path | None, file_name: str):
    if in_memory:
        uri = BytesIO()
    else:
        output_dir = Path(get_config_default("scratch_dir", scratch_dir, warn=False))
        uri = output_dir / file_name
    return uri


def extract_kwargs(callable: Callable, params: dict, kwargs: dict) -> None:
    args_spec = inspect.getfullargspec(callable)
    for arg in args_spec.args:
        if arg in kwargs:
            params[arg] = kwargs.pop(arg)
