from __future__ import annotations

try:
    import xarray as xr
except Exception as exc:
    raise ImportError(
        "Could not patch plotting API onto xarray. xarray could not be imported."
    ) from exc

from .core import stream


def patch(name="streamjoy"):
    class StreamAccessor:
        def __init__(self, resources: xr.Dataset | xr.DataArray):
            self._resources = resources

        def __call__(self, *args, **kwargs):
            return stream(self._resources, *args, **kwargs)

    StreamAccessor.__doc__ = stream.__doc__

    xr.register_dataset_accessor(name)(StreamAccessor)
    xr.register_dataarray_accessor(name)(StreamAccessor)


patch()
