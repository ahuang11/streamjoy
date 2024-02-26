from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .models import Mp4Stream, GifStream


def stream(
    resources: Any,
    path: str | Path | None = None,
    renderer: Callable | None = None,
    **kwargs,
) -> None:
    import xarray as xr
    import holoviews as hv
    
    stream_type = Mp4Stream
    if path:
        path = Path(path)
        if path.suffix == ".mp4":
            stream_type = Mp4Stream
        elif path.suffix == ".gif":
            stream_type = GifStream
        else:
            raise ValueError(f"Unsupported file extension {path.suffix}")

    params = {"renderer": renderer}
    params.update(
        {key: kwargs.pop(key) for key in stream_type.param.values() if key in kwargs}
    )
    if isinstance(resources, (xr.Dataset, xr.DataArray)):
        stream = stream_type.from_xarray(resources, **params)
    elif isinstance(resources, (hv.DynamicMap, hv.HoloMap)):
        stream = stream_type.from_holoviews(resources, **params)
    elif isinstance(resources, str) and "://" in resources:
        stream = stream_type.from_url(resources, **params)
    else:
        stream = stream_type(resources, **params)

    if path:
        return stream.write(path, **kwargs)
    return stream