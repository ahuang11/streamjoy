import re
from typing import Any

from . import _utils


def default_pandas_renderer(df_sub: "pd.DataFrame", *args, **kwargs):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    title = kwargs.get("title")
    if title and "{" in title and "}" in title:
        # find the first item in {}
        index = re.search(r"\{(.*?)\}", title).group(1)
        if df_sub.index.name == index:
            title = df_sub.index[-1]
        else:
            title = df_sub[index].iloc[-1]
    elif title is None:
        title = df_sub.index[-1]
    kwargs["title"] = title

    df_sub = df_sub.reset_index()
    groupby = kwargs.pop("groupby", None)
    if groupby:
        for group, df_group in df_sub.groupby(groupby):
            df_group.plot(*args, ax=ax, label=group, **kwargs)
    else:
        df_sub.plot(*args, ax=ax, **kwargs)

    return fig


def default_xarray_renderer(da_sel: "xr.DataArray", *args, **kwargs):
    import matplotlib.pyplot as plt

    da_sel = _utils.validate_xarray(da_sel, warn=False)

    fig = plt.figure()
    ax = plt.axes(**kwargs.pop("subplot_kws", {}))
    try:
        da_sel.plot(ax=ax, extend="both", *args, **kwargs)
    except Exception:
        da_sel.plot(ax=ax, *args, **kwargs)
    return fig


def default_holoviews_renderer(hv_obj: "hv.Element", *args, **kwargs):
    clims = kwargs.pop("clims", {})

    for hv_el in hv_obj.traverse(full_breadth=False):
        try:
            vdim = hv_el.vdims[0].name
        except IndexError:
            continue
        if vdim in clims:
            hv_el.opts(clim=clims[vdim])

    hv_obj.opts(toolbar=None, **kwargs)
    return hv_obj
