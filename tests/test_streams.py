import pytest
from imageio.v3 import improps

from streamjoy.models import Paused
from streamjoy.streams import GifStream, Mp4Stream
from streamjoy.wrappers import wrap_matplotlib


class AbstractTestMediaStream:
    def _assert_stream_and_props(self, sj, stream_cls):
        assert isinstance(sj, stream_cls)
        buf = sj.write()
        props = improps(buf)
        props.n_images == 3

    def test_from_pandas(self, stream_cls, df):
        sj = stream_cls.from_pandas(df)
        self._assert_stream_and_props(sj, stream_cls)

    def test_from_polars(self, stream_cls, pl_df):
        sj = stream_cls.from_polars(pl_df)
        self._assert_stream_and_props(sj, stream_cls)

    def test_from_xarray(self, stream_cls, ds):
        sj = stream_cls.from_xarray(ds)
        self._assert_stream_and_props(sj, stream_cls)

    def test_from_holoviews_hmap(self, stream_cls, hmap):
        sj = stream_cls.from_holoviews(hmap)
        self._assert_stream_and_props(sj, stream_cls)

    def test_from_holoviews_dmap(self, stream_cls, dmap):
        sj = stream_cls.from_holoviews(dmap)
        self._assert_stream_and_props(sj, stream_cls)

    def test_from_url_dir(self, stream_cls):
        sj = stream_cls.from_url(
            "https://noaadata.apps.nsidc.org/NOAA/G02135/north/daily/images/1978/10_Oct/",
            pattern="N_197810*_conc_v3.0.png",
        )
        self._assert_stream_and_props(sj, stream_cls)

    def test_from_url_path(self, stream_cls):
        sj = stream_cls.from_url(
            "https://github.com/ahuang11/streamjoy/raw/main/tests/data/gapminder.parquet",
        )
        self._assert_stream_and_props(sj, stream_cls)

    def test_from_directory(self, stream_cls, data_dir):
        sj = stream_cls.from_directory(data_dir, pattern="*.png")
        self._assert_stream_and_props(sj, stream_cls)

    def test_fsspec_fs(self, stream_cls, df, fsspec_fs):
        sj = stream_cls.from_pandas(df, fsspec_fs=fsspec_fs)
        self._assert_stream_and_props(sj, stream_cls)


class TestGifStream(AbstractTestMediaStream):
    @pytest.fixture(scope="class")
    def stream_cls(self):
        return GifStream

    def test_paused(self, stream_cls, df):
        @wrap_matplotlib()
        def renderer(df, groupby=None):  # TODO: fix bug groupby not needed
            return Paused(df.plot(), seconds=2)

        buf = stream_cls.from_pandas(df, renderer=renderer).write()
        props = improps(buf)
        assert props.n_images == 3


class TestMp4Stream(AbstractTestMediaStream):
    @pytest.fixture(scope="class")
    def stream_cls(self):
        return Mp4Stream

    def test_paused(self, stream_cls, df):
        @wrap_matplotlib()
        def renderer(df, groupby=None):  # TODO: fix bug groupby not needed
            return Paused(df.plot(), seconds=2)

        buf = stream_cls.from_pandas(df, renderer=renderer).write()
        props = improps(buf)
        assert props.n_images == 9
