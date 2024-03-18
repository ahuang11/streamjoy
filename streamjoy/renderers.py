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
    try:
        da_sel.plot(ax=ax, extend="both", *args, **kwargs)
    except Exception:
        da_sel.plot(ax=ax, *args, **kwargs)
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

    backend = kwargs.get("backend", "bokeh")
    hv.extension(backend)

    clims = kwargs.pop("clims", {})
    for hv_el in hv_obj.traverse(full_breadth=False):
        try:
            vdim = hv_el.vdims[0].name
        except IndexError:
            continue
        if vdim in clims:
            hv_el.opts(clim=clims[vdim])

    if backend == "bokeh":
        kwargs["toolbar"] = None
    hv_obj.opts(**kwargs)

    return hv_obj
