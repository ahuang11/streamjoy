import holoviews as hv
import matplotlib.pyplot as plt
import pytest

from streamjoy.renderers import (
    default_holoviews_renderer,
    default_pandas_renderer,
    default_polars_renderer,
    default_xarray_renderer,
)


class TestDefaultRenderer:
    @pytest.mark.parametrize("title", ["Constant", "{Year}", None])
    def test_pandas(self, df, title):
        fig = default_pandas_renderer(
            df, x="Year", y="life", groupby="Country", title=title
        )
        assert isinstance(fig, plt.Figure)

    @pytest.mark.parametrize("title", ["Constant", "{Year}", None])
    def test_polars(self, pl_df, title):
        rendered_obj = default_polars_renderer(
            pl_df, x="Year", y="life", groupby="Country", title=title
        )
        assert isinstance(rendered_obj, hv.NdOverlay)

    @pytest.mark.parametrize("title", ["Constant", "{time}", None])
    def test_xarray(self, ds, title):
        da = ds.air.isel(time=0)
        fig = default_xarray_renderer(da, title=title)
        assert isinstance(fig, plt.Figure)

    @pytest.mark.parametrize("title", ["Constant", "{x}", None])
    def test_holoviews(self, title):
        hv_obj = hv.Curve([1, 2, 3])
        rendered_obj = default_holoviews_renderer(hv_obj, title=title)
        assert isinstance(rendered_obj, hv.Element)
