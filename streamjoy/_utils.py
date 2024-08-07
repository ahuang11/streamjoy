from __future__ import annotations

import inspect
import logging
import os
from collections.abc import Iterable
from io import BytesIO
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import imageio.v3 as iio
import numpy as np
import param
from dask.distributed import Client, Future, get_client

from .models import Paused
from .settings import config

if TYPE_CHECKING:
    try:
        import xarray as xr
    except ImportError:
        xr = None

    try:
        from selenium.webdriver.remote.webdriver import BaseWebDriver
    except ImportError:
        BaseWebDriver = None


def update_logger(
    level: str | None = None,
    format: str | None = None,
    datefmt: str | None = None,
) -> logging.Logger:
    success_level = config["logging_success_level"]

    class CustomLogger(logging.Logger):
        def success(self, message, *args, **kws):
            if self.isEnabledFor(success_level):
                self._log(success_level, message, args, **kws)

    color = config["logging_success_color"]
    reset = config["logging_reset_color"]
    logging.setLoggerClass(CustomLogger)
    logging.addLevelName(success_level, f"{color}SUCCESS{reset}")
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
        f"No {color}{key!r}{reset} specified; using the default "
        f"{color}{default_value!r}{reset}"
    )
    if total_value is not None:
        message += f" / {total_value!r}"
    if suffix:
        message += f" {suffix}"
    message += f". Suppress this by passing {key!r}."

    if total_value is not None and default_value < total_value:
        logging.warning(message)
    elif total_value is None:
        logging.warning(message)


def get_config_default(
    key: str,
    value: Any,
    warn: bool = True,
    require: bool = True,
    config_prefix: str = "",
    **warn_kwargs,
) -> Any:
    config_alias = f"{config_prefix}_{key}" if config_prefix else key
    if require and config_alias not in config:
        raise ValueError(f"Missing required config key: {config_alias}")

    if value is None:
        default = config[config_alias]
        if warn:
            value = warn_default_used(key, default, **warn_kwargs)
        value = default
    return value


def populate_config_defaults(
    params: dict[str, Any],
    keys: param.Parameterized,
    warn_on: list[str] | None = None,
    config_prefix: str = "",
):
    for key in keys:
        config_alias = f"{config_prefix}_{key}" if config_prefix else key
        if config_alias not in config.keys():
            continue
        warn = key in (warn_on or [])
        value = get_config_default(
            key, params.get(key), warn=warn, config_prefix=config_prefix
        )
        params[key] = value
    params = {key: value for key, value in params.items() if value is not None}
    return params


def get_distributed_client(client: Client | None = None, **kwargs) -> Client:
    if client is not None:
        return client

    try:
        client = get_client()
    except ValueError:
        client = Client(**kwargs)
    return client


def download_file(
    url: str,
    scratch_dir: Path | None = None,
    in_memory: bool = False,
    parent_depth: int | None = None,
) -> str:
    try:
        import requests
    except ImportError:
        raise ImportError("To directly read from a URL, `pip install requests`")

    url_path = Path(url)
    file_name = url_path.name

    parent_depth = get_config_default("parent_depth", parent_depth, warn=False)
    for _ in range(parent_depth):
        url_path = url_path.parent
        file_name = f"{url_path.name}_{file_name}"
    uri = resolve_uri(file_name=file_name, scratch_dir=scratch_dir, in_memory=in_memory)
    if not isinstance(uri, BytesIO) and os.path.exists(uri):
        return uri

    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(uri, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return uri


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
    elif max_frames > total_frames:
        max_frames = total_frames
    elif max_frames == -1:
        max_frames = total_frames
    return max_frames


def get_first(iterable):
    if isinstance(iterable, (list, tuple)):
        return iterable[0]
    return next(islice(iterable, 0, 1), None)


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


def resolve_uri(
    file_name: str | None = None,
    scratch_dir: str | Path | None = None,
    in_memory: bool = False,
    fsspec_fs: Any | None = None,
) -> str | Path | BytesIO:
    if in_memory:
        return BytesIO()

    output_dir = get_config_default("scratch_dir", scratch_dir, warn=False)
    if fsspec_fs:
        try:
            fsspec_fs.mkdir(output_dir, exist_ok=True, parents=True)
        except FileExistsError:
            pass
        uri = os.path.join(output_dir, file_name)
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        uri = output_dir / file_name
    return uri


def pop_kwargs(callable: Callable, kwargs: dict) -> None:
    args_spec = inspect.getfullargspec(callable)
    return {arg: kwargs.pop(arg) for arg in args_spec.args if arg in kwargs}


def pop_from_cls(cls: type, kwargs: dict) -> dict:
    return {
        key: kwargs.pop(key) for key in set(kwargs) if key not in cls.param.values()
    }


def import_function(import_path: str) -> Callable:
    module, function = import_path.rsplit(".", 1)
    module = __import__(module, fromlist=[function])
    return getattr(module, function)


def validate_xarray(
    ds: xr.Dataset | xr.DataArray,
    dim: str | None = None,
    var: str | None = None,
    warn: bool = True,
    raise_ndim: bool = True,
):
    import xarray as xr

    if var:
        ds = ds[var]
    elif isinstance(ds, xr.Dataset):
        var = list(ds.data_vars)[0]
        if warn:
            warn_default_used("var", var, suffix="from the dataset")
        ds = ds[var]

    squeeze_dims = [d for d in ds.dims if d != dim and ds.sizes[d] == 1]
    ds = ds.squeeze(squeeze_dims)
    if ds.ndim > 3 and raise_ndim:
        raise ValueError(f"Can only handle 3D arrays; {ds.ndim}D array found")
    return ds


def validate_renderer_iterables(
    resources: list[Any],
    iterables: list[list[Any]],
):
    if iterables is None:
        return

    num_iterables = len(iterables)
    num_resources = len(resources)

    if num_iterables == num_resources:
        logging.warning(
            "The length of the iterables matches the length of the resources. "
            "This is likely not what you want; the iterables should be a list of lists, "
            "where each inner list corresponds to the arguments for each frame."
        )

    if not isinstance(iterables[0], Iterable) or isinstance(iterables[0], str):
        raise TypeError(
            "Iterables should be like a list of lists, where each inner list corresponds "
            "to the arguments for each frame; e.g. `[[arg1_for_frame1, arg1_for_frame2], "
            "[arg2_for_frame_1, arg2_for_frame2]]`"
        )


def map_over(client, func, resources, batch_size, *args, **kwargs):
    try:
        return client.map(func, resources, *args, batch_size=batch_size, **kwargs)
    except TypeError:
        return [
            client.submit(func, resource, *args, **kwargs) for resource in resources
        ]


def repeat_frame(
    write: Callable, image: np.ndarray, seconds: int, fps: int, **write_kwargs
) -> np.ndarray:
    if seconds == 0:
        return image

    repeat = int(seconds * fps)
    for _ in range(repeat):
        write(image, **write_kwargs)
    return image


def imread_with_pause(
    uri: Any | Paused,
    extension: str | None = None,
    plugin: str | None = None,
    fsspec_fs: Any | None = None,
) -> np.ndarray | Paused:
    imread_kwargs = dict(extension=extension, plugin=plugin)
    seconds = None
    if isinstance(uri, Paused):
        seconds = uri.seconds
        uri = uri.output
    if fsspec_fs:
        with fsspec_fs.open(uri, "rb") as f:
            image = iio.imread(f, **imread_kwargs)
    else:
        image = iio.imread(uri, **imread_kwargs).squeeze()
    image = image.squeeze()
    if seconds is None:
        return image
    else:
        return Paused(output=image, seconds=seconds)


def subset_resources_renderer_iterables(
    resources: Any, renderer_iterables: list[Any], max_frames: int
):
    resources = resources[: max_frames or max_frames]
    renderer_iterables = [
        iterable[: len(resources)] for iterable in renderer_iterables or []
    ]
    return resources, renderer_iterables


def get_webdriver_path(webdriver: str):
    if webdriver.lower() == "chrome":
        from webdriver_manager.chrome import ChromeDriverManager

        webdriver_path = ChromeDriverManager().install()
    elif webdriver.lower() == "firefox":
        from webdriver_manager.firefox import GeckoDriverManager

        webdriver_path = GeckoDriverManager().install()
    return webdriver_path


def get_webdriver(webdriver: tuple[str, str] | Callable) -> BaseWebDriver:
    if isinstance(webdriver, Callable):
        return webdriver()

    webdriver_key, webdriver_path = webdriver
    if webdriver_key.lower() == "chrome":
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.webdriver import Service, WebDriver

        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-extensions")
        webdriver_path = webdriver_path or get_webdriver_path("chrome")
        driver = WebDriver(service=Service(webdriver_path), options=options)

    elif webdriver_key.lower() == "firefox":
        from selenium.webdriver.firefox.options import Options
        from selenium.webdriver.firefox.webdriver import Service, WebDriver

        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-extensions")
        webdriver_path = webdriver_path or get_webdriver_path("firefox")
        driver = WebDriver(service=Service(webdriver_path), options=options)

    else:
        raise NotImplementedError(
            f"Webdriver {webdriver_key} not supported; "
            f"use 'chrome' or 'firefox', or pass a custom callable."
        )

    return driver
