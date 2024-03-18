from io import BytesIO

import streamjoy.pandas  # noqa: F401


class TestPandas:
    def test_dataframe(self, df):
        stream = df.streamjoy(groupby="Country")
        assert stream.renderer_kwargs == {
            "groupby": "Country",
            "x": "Year",
            "y": "fertility",
            "xlabel": "Year",
            "ylabel": "Fertility",
        }
        assert isinstance(stream.write(), BytesIO)

    def test_series(self, df):
        stream = df.set_index("Year")[["Country", "life"]].streamjoy(groupby="Country")
        assert stream.renderer_kwargs == {
            "groupby": "Country",
            "x": "Year",
            "y": "life",
            "xlabel": "Year",
            "ylabel": "Life",
        }
        assert isinstance(stream.write(), BytesIO)
