from __future__ import annotations

import logging
from inspect import isgenerator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import holoviews as hv
import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from . import _utils
from .models import Serialized
from .renderers import (
    default_holoviews_renderer,
    default_pandas_renderer,
    default_polars_renderer,
    default_xarray_renderer,
)
from .settings import file_handlers, obj_handlers
from .wrappers import wrap_holoviews, wrap_matplotlib

if TYPE_CHECKING:
    from .streams import MediaStream


def _select_obj_handler(resources: Any) -> MediaStream:
    if isinstance(resources, str) and "://" in resources:
        return serialize_url
    if isinstance(resources, (Path, str)):
        return serialize_paths

    for class_or_package_name, function_name in obj_handlers.items():
        module = getattr(resources, "__module__", "").split(".", maxsplit=1)[0]
        type_ = type(resources).__name__
        if (
            f"{module}.{type_}" == class_or_package_name
            or module == class_or_package_name
        ):
            return globals()[function_name]

    raise ValueError(
        f"Could not find a method to handle {type(resources)}; "
        f"supported classes/packages are {list(obj_handlers.keys())}."
    )


def serialize_xarray(
    stream_cls,
    resources: xr.Dataset | xr.DataArray,
    renderer: Callable | None = None,
    renderer_iterables: list[Any] | None = None,
    renderer_kwargs: dict | None = None,
    **kwargs,
) -> Serialized:
    """
    Serialize xarray datasets or data arrays for streaming or rendering.

    Args:
        stream_cls: The class reference used for logging and utility functions.
        resources: The xarray dataset or data array to be serialized.
        renderer: The rendering function to use on the dataset.
        renderer_iterables: Additional iterable arguments to pass to the renderer.
        renderer_kwargs: Additional keyword arguments to pass to the renderer.
        **kwargs: Additional keyword arguments, including 'dim' and 'var' for xarray selection.

    Returns:
        A tuple containing the serialized resources, renderer, renderer_iterables, renderer_kwargs, and any additional keyword arguments.
    """

    ds = resources
    dim = kwargs.pop("dim", None)
    var = kwargs.pop("var", None)

    ds = _utils.validate_xarray(ds, dim=dim, var=var)
    if not dim:
        dim = list(ds.dims)[0]
        _utils.warn_default_used("dim", dim, suffix="from the dataset")
    elif dim not in ds.dims:
        raise ValueError(f"{dim!r} not in {ds.dims!r}")

    total_frames = len(ds[dim])
    max_frames = _utils.get_max_frames(total_frames, kwargs.get("max_frames"))
    resources = [ds.isel({dim: i}) for i in range(max_frames)]

    renderer_kwargs = renderer_kwargs or {}
    renderer_kwargs.update(_utils.pop_from_cls(stream_cls, kwargs))

    if renderer is None:
        renderer = wrap_matplotlib(
            in_memory=kwargs.get("in_memory"),
            scratch_dir=kwargs.get("scratch_dir"),
        )(default_xarray_renderer)
        ds_0 = resources[0]
        if ds_0.ndim >= 2:
            renderer_kwargs["vmin"] = renderer_kwargs.get(
                "vmin", _utils.get_result(ds_0.min()).item()
            )
            renderer_kwargs["vmax"] = renderer_kwargs.get(
                "vmax", _utils.get_result(ds_0.max()).item()
            )
        else:
            renderer_kwargs["ylim"] = renderer_kwargs.get(
                "ylim",
                (
                    _utils.get_result(ds_0.min()).item(),
                    _utils.get_result(ds_0.max()).item(),
                ),
            )
    return Serialized(resources, renderer, renderer_iterables, renderer_kwargs, kwargs)


def serialize_pandas(
    stream_cls,
    resources: pd.DataFrame,
    renderer: Callable | None = None,
    renderer_iterables: list[Any] | None = None,
    renderer_kwargs: dict | None = None,
    **kwargs,
) -> Serialized:
    """
    Serialize pandas DataFrame for streaming or rendering.

    Args:
        stream_cls: The class reference used for logging and utility functions.
        resources: The pandas DataFrame to be serialized.
        renderer: The rendering function to use on the DataFrame.
        renderer_iterables: Additional iterable arguments to pass to the renderer.
        renderer_kwargs: Additional keyword arguments to pass to the renderer.
        **kwargs: Additional keyword arguments, including 'groupby' for DataFrame grouping.

    Returns:
        A tuple containing the serialized resources, renderer, renderer_iterables, renderer_kwargs, and any additional keyword arguments.
    """
    import pandas as pd

    df = resources
    groupby = kwargs.get("groupby")

    total_frames = df.groupby(groupby).size().max() if groupby else len(df)
    max_frames = _utils.get_max_frames(total_frames, kwargs.get("max_frames"))
    resources = [
        df.groupby(groupby, as_index=False).head(i) if groupby else df.head(i)
        for i in range(1, max_frames + 1)
    ]

    renderer_kwargs = renderer_kwargs or {}
    renderer_kwargs.update(_utils.pop_from_cls(stream_cls, kwargs))

    if renderer is None:
        renderer = wrap_matplotlib(
            in_memory=kwargs.get("in_memory"),
            scratch_dir=kwargs.get("scratch_dir"),
        )(default_pandas_renderer)
        if "x" not in renderer_kwargs:
            if df.index.name or isinstance(df, pd.Series):
                renderer_kwargs["x"] = df.index.name
            else:
                for col in df.columns:
                    if col != groupby:
                        break
                renderer_kwargs["x"] = col
            _utils.warn_default_used(
                "x", renderer_kwargs["x"], suffix="from the dataframe"
            )
        if "y" not in renderer_kwargs:
            if isinstance(df, pd.Series):
                col = df.name
            else:
                numeric_cols = df.select_dtypes(include="number").columns
                for col in numeric_cols:
                    if col not in (renderer_kwargs["x"], groupby):
                        break
            renderer_kwargs["y"] = col
            _utils.warn_default_used(
                "y", renderer_kwargs["y"], suffix="from the dataframe"
            )
        if "xlabel" not in renderer_kwargs:
            renderer_kwargs["xlabel"] = renderer_kwargs["x"].title().replace("_", " ")
        if "ylabel" not in renderer_kwargs:
            renderer_kwargs["ylabel"] = renderer_kwargs["y"].title().replace("_", " ")

    return Serialized(resources, renderer, renderer_iterables, renderer_kwargs, kwargs)


def serialize_polars(
    stream_cls,
    resources: pl.DataFrame,
    renderer: Callable | None = None,
    renderer_iterables: list[Any] | None = None,
    renderer_kwargs: dict | None = None,
    **kwargs,
) -> Serialized:
    """
    Serialize Polars DataFrame for streaming or rendering.

    Args:
        stream_cls: The class reference used for logging and utility functions.
        resources: The Polars DataFrame to be serialized.
        renderer: The rendering function to use on the DataFrame.
        renderer_iterables: Additional iterable arguments to pass to the renderer.
        renderer_kwargs: Additional keyword arguments to pass to the renderer.
        **kwargs: Additional keyword arguments, including 'groupby' for DataFrame grouping.

    Returns:
        A tuple containing the serialized resources, renderer, renderer_iterables, renderer_kwargs, and any additional keyword arguments.
    """
    import polars as pl

    groupby = kwargs.get("groupby")

    if groupby:
        group_sizes = resources.groupby(groupby).agg(pl.count()).sort(by="count")
        total_frames = group_sizes.select(pl.col("count").max()).to_numpy()[0, 0]
    else:
        total_frames = len(resources)

    max_frames = _utils.get_max_frames(total_frames, kwargs.get("max_frames"))
    resources_expanded = [
        resources.groupby(groupby).head(i) if groupby else resources.head(i)
        for i in range(1, max_frames + 1)
    ]

    renderer_kwargs = renderer_kwargs or {}
    renderer_kwargs.update(_utils.pop_from_cls(stream_cls, kwargs))

    if renderer is None:
        renderer = wrap_holoviews(
            in_memory=kwargs.get("in_memory"),
            scratch_dir=kwargs.get("scratch_dir"),
        )(default_polars_renderer)
        numeric_cols = [
            col
            for col in resources.columns
            if resources[col].dtype in [pl.Float64, pl.Int64, pl.Float32, pl.Int32]
        ]
        if "x" not in renderer_kwargs:
            for col in numeric_cols:
                if col != groupby:
                    renderer_kwargs["x"] = col
                    break
            _utils.warn_default_used(
                "x", renderer_kwargs["x"], suffix="from the dataframe"
            )
        if "y" not in renderer_kwargs:
            for col in numeric_cols:
                if col not in (renderer_kwargs["x"], groupby):
                    renderer_kwargs["y"] = col
                    break
            _utils.warn_default_used(
                "y", renderer_kwargs["y"], suffix="from the dataframe"
            )
        if "xlabel" not in renderer_kwargs:
            renderer_kwargs["xlabel"] = renderer_kwargs["x"].title().replace("_", " ")
        if "ylabel" not in renderer_kwargs:
            renderer_kwargs["ylabel"] = renderer_kwargs["y"].title().replace("_", " ")

    return Serialized(
        resources_expanded, renderer, renderer_iterables, renderer_kwargs, kwargs
    )


def serialize_holoviews(
    stream_cls,
    resources: hv.HoloMap | hv.DynamicMap,
    renderer: Callable | None = None,
    renderer_iterables: list[Any] | None = None,
    renderer_kwargs: dict | None = None,
    **kwargs,
) -> Serialized:
    """
    Serialize HoloViews objects for streaming or rendering.

    Args:
        stream_cls: The class reference used for logging and utility functions.
        resources: The HoloViews object to be serialized.
        renderer: The rendering function to use on the HoloViews object.
        renderer_iterables: Additional iterable arguments to pass to the renderer.
        renderer_kwargs: Additional keyword arguments to pass to the renderer.
        **kwargs: Additional keyword arguments for HoloViews object customization.

    Returns:
        A tuple containing the serialized resources, renderer, renderer_iterables, renderer_kwargs, and any additional keyword arguments.
    """
    import holoviews as hv

    backend = hv.Store.current_backend
    hv.extension(backend)

    def _select_element(hv_obj, key):
        try:
            resource = hv_obj[key]
        except Exception:
            resource = hv_obj.select(**{kdims[0].name: key})
        return resource

    hv_obj = resources
    if isinstance(hv_obj, (hv.core.spaces.DynamicMap, hv.core.spaces.HoloMap)):
        hv_map = hv_obj
    elif issubclass(
        type(hv_obj), (hv.core.layout.Layoutable, hv.core.overlay.Overlayable)
    ):
        hv_map = hv_obj[0]

    if not isinstance(hv_map, (hv.core.spaces.DynamicMap, hv.core.spaces.HoloMap)):
        raise ValueError("Can only handle HoloMap and DynamicMap objects.")
    elif isinstance(hv_map, hv.core.spaces.DynamicMap):
        kdims = hv_map.kdims
        keys = hv_map.kdims[0].values
    else:
        kdims = hv_map.kdims
        keys = hv_map.keys()

    if len(kdims) > 1:
        raise ValueError("Can only handle 1D HoloViews objects.")

    resources = [_select_element(hv_obj, key).opts(title=str(key)) for key in keys]

    renderer_kwargs = renderer_kwargs or {}
    renderer_kwargs.update(_utils.pop_from_cls(stream_cls, kwargs))

    if renderer is None:
        renderer = wrap_holoviews(
            in_memory=kwargs.get("in_memory"),
            scratch_dir=kwargs.get("scratch_dir"),
        )(default_holoviews_renderer)
        clims = {}
        for hv_el in hv_obj.traverse(full_breadth=False):
            if isinstance(hv_el, hv.DynamicMap):
                hv.render(hv_el, backend=backend)

            if isinstance(hv_el, hv.Element):
                if hv_el.ndims > 1:
                    vdim = hv_el.vdims[0].name
                    array = hv_el.dimension_values(vdim)
                    clim = (np.nanmin(array), np.nanmax(array))
                    clims[vdim] = clim
        renderer_kwargs.update(
            backend=backend,
            clims=clims,
        )

    if kwargs.get("processes"):
        logging.warning(
            "HoloViews rendering does not support processes; "
            "setting processes=False."
        )
    kwargs["processes"] = False
    return Serialized(resources, renderer, renderer_iterables, renderer_kwargs, kwargs)


def serialize_url(
    stream_cls,
    resources: str,
    renderer: Callable | None = None,
    renderer_iterables: list[Any] | None = None,
    renderer_kwargs: dict | None = None,
    **kwargs,
) -> Serialized:
    """
    Serialize resources from a URL for streaming or rendering.

    Args:
        stream_cls: The class reference used for logging and utility functions.
        resources: The URL of the resources to be serialized.
        renderer: The rendering function to use on the resources.
        renderer_iterables: Additional iterable arguments to pass to the renderer.
        renderer_kwargs: Additional keyword arguments to pass to the renderer.
        **kwargs: Additional keyword arguments, including 'pattern', 'sort_key', 'max_files', 'file_handler', and 'file_handler_kwargs'.

    Returns:
        A MediaStream object containing the serialized resources.
    """
    import re

    import requests
    from bs4 import BeautifulSoup

    base_url = resources
    pattern = kwargs.pop("pattern", None)
    sort_key = kwargs.pop("sort_key", None)
    max_files = kwargs.pop("max_files", None)
    file_handler = kwargs.pop("file_handler", None)
    file_handler_kwargs = kwargs.pop("file_handler_kwargs", None)

    response = requests.get(resources)
    content_type = response.headers.get("Content-Type")

    if pattern is not None:
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        href = re.compile(pattern.replace("*", ".*"))
        links = soup.find_all("a", href=href)
    else:
        if content_type.startswith("text"):
            raise ValueError(
                f"A pattern must be provided if the URL is a directory of files; "
                f"got {resources!r}."
            )
        links = [{"href": ""}]

    max_files = _utils.get_config_default(
        "max_files", max_files, total_value=len(links), suffix="links"
    )
    if max_files > 0:
        links = links[:max_files]

    if len(links) == 0:
        raise ValueError(f"No links found with pattern {pattern!r} at {base_url!r}.")

    # download files
    urls = [base_url + link.get("href") for link in links]
    client = _utils.get_distributed_client(
        client=kwargs.get("client"),
        processes=kwargs.get("processes"),
        threads_per_worker=kwargs.get("threads_per_worker"),
    )

    futures = _utils.map_over(
        client,
        _utils.download_file,
        urls,
        kwargs.get("batch_size"),
        scratch_dir=kwargs.get("scratch_dir"),
        in_memory=kwargs.get("in_memory"),
    )
    paths = client.gather(futures)
    return serialize_paths(
        stream_cls,
        paths,
        sort_key=sort_key,
        max_files=max_files,
        file_handler=file_handler,
        file_handler_kwargs=file_handler_kwargs,
        renderer=renderer,
        renderer_iterables=renderer_iterables,
        renderer_kwargs=renderer_kwargs,
        **kwargs,
    )


def serialize_paths(
    stream_cls,
    resources: list[str | Path] | str,
    renderer: Callable | None = None,
    renderer_iterables: list[Any] | None = None,
    renderer_kwargs: dict | None = None,
    **kwargs,
) -> Serialized:
    """
    Serialize resources from file paths for streaming or rendering.

    Args:
        resources: A list of file paths or a single file path string to be serialized.
        renderer: The rendering function to use on the resources.
        renderer_iterables: Additional iterable arguments to pass to the renderer.
        renderer_kwargs: Additional keyword arguments to pass to the renderer.
        **kwargs: Additional keyword arguments, including 'sort_key', 'max_files', 'file_handler', and 'file_handler_kwargs'.

    Returns:
        A MediaStream object containing the serialized resources.
    """
    if isinstance(resources, str):
        resources = [resources]

    sort_key = kwargs.pop("sort_key", None)
    max_files = kwargs.pop("max_files", None)
    file_handler = kwargs.pop("file_handler", None)
    file_handler_kwargs = kwargs.pop("file_handler_kwargs", None)

    paths = sorted(resources, key=sort_key)

    max_files = _utils.get_config_default(
        "max_files", max_files, total_value=len(paths), suffix="paths"
    )
    if max_files > 0:
        paths = paths[:max_files]

    # find a file handler
    extension = Path(paths[0]).suffix
    file_handler_meta = file_handlers.get(extension, {})
    file_handler_import_path = file_handler_meta.get("import_path")
    file_handler_concat_path = file_handler_meta.get("concat_path")
    if file_handler is None and file_handler_import_path is not None:
        file_handler = _utils.import_function(file_handler_import_path)
        if file_handler_concat_path is not None:
            file_handler_concat = _utils.import_function(file_handler_concat_path)

    # read as objects
    if file_handler is not None:
        if file_handler_concat_path is not None:
            resources = file_handler_concat(
                file_handler(path, **(file_handler_kwargs or {})) for path in paths
            )
        else:
            resources = file_handler(paths, **(file_handler_kwargs or {}))
        return serialize_appropriately(
            stream_cls,
            resources,
            renderer,
            renderer_iterables,
            renderer_kwargs,
            **kwargs,
        )

    # or simply return image paths
    return Serialized(paths, renderer, renderer_iterables, renderer_kwargs, kwargs)


def serialize_appropriately(
    stream_cls,
    resources: Any,
    renderer: Callable,
    renderer_iterables: list[Any],
    renderer_kwargs: dict[str, Any],
    **kwargs,
) -> Serialized:
    """
    Automatically select the appropriate serialization method based on the type of resources.

    Args:
        stream_cls: The class reference used for logging and utility functions.
        resources: The resources to be serialized, which can be of any type.
        renderer: The rendering function to use on the resources.
        renderer_iterables: Additional iterable arguments to pass to the renderer.
        renderer_kwargs: Additional keyword arguments to pass to the renderer.
        **kwargs: Additional keyword arguments for further customization.

    Returns:
        A Serialized object containing the serialized resources.
    """
    if not (isinstance(resources, (list, tuple)) or isgenerator(resources)):
        obj_handler = _select_obj_handler(resources)
        _utils.validate_renderer_iterables(resources, renderer_iterables)

        serialized = obj_handler(
            stream_cls,
            resources,
            renderer=renderer,
            renderer_iterables=renderer_iterables,
            renderer_kwargs=renderer_kwargs,
            **kwargs,
        )
        resources = serialized.resources
        renderer = serialized.renderer
        renderer_iterables = serialized.renderer_iterables
        renderer_kwargs = serialized.renderer_kwargs
        kwargs = serialized.kwargs
        max_frames = _utils.get_max_frames(len(resources), kwargs.get("max_frames"))
        kwargs["max_frames"] = max_frames
        resources, renderer_iterables = _utils.subset_resources_renderer_iterables(
            resources, renderer_iterables, max_frames
        )
        _utils.pop_kwargs(obj_handler, renderer_kwargs)
        _utils.pop_kwargs(obj_handler, kwargs)
    return Serialized(resources, renderer, renderer_iterables, renderer_kwargs, kwargs)
