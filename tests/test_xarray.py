from io import BytesIO

import streamjoy.xarray  # noqa: F401


class TestXArray:
    def test_dataset_3d(self, ds):
        sj = ds.streamjoy()
        assert "vmin" in sj.renderer_kwargs
        assert "vmax" in sj.renderer_kwargs
        assert isinstance(sj.write(), BytesIO)

    def test_dataarray_3d(self, ds):
        sj = ds["air"].streamjoy()
        assert "vmin" in sj.renderer_kwargs
        assert "vmax" in sj.renderer_kwargs
        assert isinstance(sj.write(), BytesIO)

    def test_dataset_2d(self, ds):
        sj = ds.mean("lat").streamjoy()
        assert "ylim" in sj.renderer_kwargs
        assert isinstance(sj.write(), BytesIO)

    def test_dataarray_2d(self, ds):
        sj = ds["air"].mean("lat").streamjoy()
        assert "ylim" in sj.renderer_kwargs
        assert isinstance(sj.write(), BytesIO)
