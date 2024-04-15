from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import _utils

if TYPE_CHECKING:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None

    try:
        import pandas as pd
    except ImportError:
        pd = None

    try:
        import polars as pl
    except ImportError:
        pl = None

    try:
        import xarray as xr
    except ImportError:
        xr = None

    try:
        import holoviews as hv
    except ImportError:
        hv = None


def default_pandas_renderer(
    df_sub: pd.DataFrame, *args: tuple[Any], **kwargs: dict[str, Any]
) -> plt.Figure:
    """
    Render a pandas DataFrame using matplotlib.

    Args:
        df_sub: The DataFrame to render.
        *args: Additional positional arguments to pass to the renderer.
        **kwargs: Additional keyword arguments to pass to the renderer.

    Returns
        A matplotlib figure.
    """
    import matplotlib.pyplot as plt

    df_sub = df_sub.reset_index()

    fig, ax = plt.subplots()

    title = kwargs.get("title")
    if title:
        title = title.format(**df_sub.iloc[-1])
    elif title is None:
        title = df_sub[kwargs["x"]].iloc[-1]
    kwargs["title"] = title

    groupby = kwargs.pop("groupby", None)
    if groupby:
        for group, df_group in df_sub.groupby(groupby):
            df_group.plot(*args, ax=ax, label=group, **kwargs)
    else:
        df_sub.plot(*args, ax=ax, **kwargs)

    return fig


def default_polars_renderer(
    df_sub: pl.DataFrame, *args: tuple[Any], **kwargs: dict[str, Any]
) -> hv.Element:
    """
    Render a polars DataFrame using HoloViews.

    Args:
        df_sub: The DataFrame to render.
        *args: Additional positional arguments to pass to the renderer.
        **kwargs: Additional keyword arguments to pass to the renderer.

    Returns:
        The rendered HoloViews Element.
    """
    backend = kwargs.pop("backend", None)
    by = kwargs.pop("groupby", None)

    title = kwargs.get("title")
    if title:
        title = title.format(**df_sub.tail(1).to_pandas().to_dict("records")[0])
    elif title is None:
        title = df_sub[kwargs["x"]].tail(1)[0]
    kwargs["title"] = str(title)

    if by:
        kwargs["by"] = by
    hv_obj = df_sub.plot(*args, **kwargs)
    return default_holoviews_renderer(hv_obj, backend=backend)


def default_xarray_renderer(
    da_sel: xr.DataArray, *args: tuple[Any], **kwargs: dict[str, Any]
) -> plt.Figure:
    """
    Render an xarray DataArray using matplotlib.

    Args:
        da_sel: The DataArray to render.
        *args: Additional positional arguments to pass to the renderer.
        **kwargs: Additional keyword arguments to pass to the renderer.

    Returns:
        A matplotlib figure.
    """
    import matplotlib.pyplot as plt

    da_sel = _utils.validate_xarray(da_sel, warn=False)

    fig = plt.figure()
    ax = plt.axes(**kwargs.pop("subplot_kws", {}))
    title = kwargs.pop("title", None)

    try:
        da_sel.plot(ax=ax, extend="both", *args, **kwargs)
    except Exception:
        da_sel.plot(ax=ax, *args, **kwargs)

    if title:
        title_format = {coord: da_sel[coord].values for coord in da_sel.coords}
        ax.set_title(title.format(**title_format))

    return fig


def default_holoviews_renderer(
    hv_obj: hv.Element, *args: tuple[Any], **kwargs: dict[str, Any]
) -> hv.Element:
    """
    Render a HoloViews Element using the default backend.

    Args:
        hv_obj: The HoloViews Element to render.
        *args: Additional positional arguments to pass to the renderer.
        **kwargs: Additional keyword arguments to pass to the renderer.

    Returns:
        The rendered HoloViews Element.
    """
    import holoviews as hv

    backend = kwargs.get("backend", hv.Store.current_backend)

    clims = kwargs.pop("clims", {})
    for hv_el in hv_obj.traverse(full_breadth=False):
        try:
            vdim = hv_el.vdims[0].name
        except IndexError:
            continue
        if vdim in clims:
            hv_el.opts(clim=clims[vdim], backend=backend)

    if backend == "bokeh":
        kwargs["toolbar"] = None
    elif backend == "matplotlib":
        kwargs["cbar_extend"] = kwargs.get("cbar_extend", "both")

    if isinstance(hv_obj, hv.Overlay):
        for hv_el in hv_obj:
            try:
                hv_el.opts(**kwargs)
            except Exception:
                pass
    else:
        hv_obj.opts(**kwargs)

    return hv_obj
