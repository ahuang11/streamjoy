import holoviews as hv
import matplotlib.pyplot as plt

from streamjoy.renderers import (
    default_holoviews_renderer,
    default_pandas_renderer,
    default_xarray_renderer,
)


class TestDefaultRenderer:
    def test_pandas(self, df):
        fig = default_pandas_renderer(df, x="Year", y="life", groupby="Country")
        assert isinstance(fig, plt.Figure)

    def test_xarray(self, ds):
        da = ds.air.isel(time=0)
        fig = default_xarray_renderer(da)
        assert isinstance(fig, plt.Figure)

    def test_holoviews(self):
        hv_obj = hv.Curve([1, 2, 3])
        rendered_obj = default_holoviews_renderer(hv_obj)
        assert isinstance(rendered_obj, hv.Element)
