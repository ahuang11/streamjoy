from __future__ import annotations

try:
    import polars as pl
except Exception as exc:
    raise ImportError(
        "Could not patch streamjoy API onto polars. Polars could not be imported."
    ) from exc

from .core import stream


def patch(name="streamjoy"):
    class StreamAccessor:
        def __init__(self, resources: pl.DataFrame | pl.Series | pl.LazyFrame):
            self._resources = resources

        def __call__(self, *args, **kwargs):
            return stream(self._resources, *args, **kwargs)

    pl.api.register_dataframe_namespace(name)(StreamAccessor)
    pl.api.register_series_namespace(name)(StreamAccessor)
    pl.api.register_lazyframe_namespace(name)(StreamAccessor)


patch()
