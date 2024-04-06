from streamjoy.models import Serialized
from streamjoy.serializers import (
    serialize_holoviews,
    serialize_numpy,
    serialize_pandas,
    serialize_polars,
    serialize_xarray,
)


class TestSerializeNumpy:
    def test_serialize_numpy(self, array):
        serialized = serialize_numpy(None, array)
        assert isinstance(serialized, Serialized)
        assert len(serialized.resources) == 36
        assert isinstance(serialized.resources, list)
        assert not serialized.renderer
        assert serialized.renderer_iterables is None
        assert isinstance(serialized.renderer_kwargs, dict)
        assert isinstance(serialized.kwargs, dict)


class TestSerializeXarray:
    def test_serialize_xarray(self, ds):
        serialized = serialize_xarray(None, ds)
        assert isinstance(serialized, Serialized)
        assert len(serialized.resources) == 3
        assert isinstance(serialized.resources, list)
        assert callable(serialized.renderer)
        assert serialized.renderer_iterables is None
        assert isinstance(serialized.renderer_kwargs, dict)
        assert isinstance(serialized.kwargs, dict)


class TestSerializePandas:
    def test_serialize_pandas(self, df):
        serialized = serialize_pandas(None, df)
        assert isinstance(serialized, Serialized)
        assert len(serialized.resources) == 3
        assert isinstance(serialized.resources, list)
        assert callable(serialized.renderer)
        assert serialized.renderer_iterables is None
        assert isinstance(serialized.renderer_kwargs, dict)
        assert isinstance(serialized.kwargs, dict)


class TestSerializePolars:
    def test_serialize_polars(self, pl_df):
        serialized = serialize_polars(None, pl_df)
        assert isinstance(serialized, Serialized)
        assert len(serialized.resources) == 3
        assert isinstance(serialized.resources, list)
        assert callable(serialized.renderer)
        assert serialized.renderer_iterables is None
        assert isinstance(serialized.renderer_kwargs, dict)
        assert isinstance(serialized.kwargs, dict)


class TestSerializeHoloviews:
    def test_serialize_holoviews(self, hmap):
        serialized = serialize_holoviews(None, hmap)
        assert isinstance(serialized, Serialized)
        assert len(serialized.resources) == 20
        assert isinstance(serialized.resources, list)
        assert callable(serialized.renderer)
        assert serialized.renderer_iterables is None
        assert isinstance(serialized.renderer_kwargs, dict)
        assert isinstance(serialized.kwargs, dict)
