import pytest
from imageio.v3 import improps

from streamjoy.streams import GifStream, Mp4Stream


class AbstractTestMediaStream:
    def _assert_stream_and_props(self, sj, stream_cls):
        assert isinstance(sj, stream_cls)
        buf = sj.write()
        props = improps(buf)
        props.n_images == 3

    def test_from_pandas(self, stream_cls, df):
        sj = stream_cls.from_pandas(df)
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


class TestGifStream(AbstractTestMediaStream):
    @pytest.fixture(scope="class")
    def stream_cls(self):
        return GifStream


class TestMp4Stream(AbstractTestMediaStream):
    @pytest.fixture(scope="class")
    def stream_cls(self):
        return Mp4Stream
