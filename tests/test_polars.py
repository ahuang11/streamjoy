from io import BytesIO

import streamjoy.polars  # noqa: F401


class TestPolars:
    def test_dataframe(self, pl_df):
        stream = pl_df.streamjoy(groupby="Country")
        assert stream.renderer_kwargs == {
            "groupby": "Country",
            "x": "Year",
            "y": "fertility",
            "xlabel": "Year",
            "ylabel": "Fertility",
        }
        assert isinstance(stream.write(), BytesIO)
