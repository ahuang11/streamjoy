from io import BytesIO
from pathlib import Path

import pytest

from streamjoy.core import connect, stream
from streamjoy.streams import AnyStream, ConnectedStreams, GifStream, Mp4Stream


class TestStream:
    def test_no_uri(self):
        result = stream(resources=[0, 1, 2])
        assert isinstance(
            result, AnyStream
        ), "Expected an instance of AnyStream or its subclass"

    def test_uri(self, df, tmp_path):
        uri = tmp_path / "test.mp4"
        stream(resources=df, uri=uri)
        assert uri.exists(), "Stream file should exist"

    def test_uri_with_bytesio(self, df):
        uri = BytesIO()
        result = stream(resources=df, uri=uri, extension=".mp4")
        assert isinstance(
            result, BytesIO
        ), "Expected a BytesIO object when URI is a BytesIO instance"

    def test_uri_with_str_path(self, df, tmp_path):
        uri = str(tmp_path / "test.gif")
        stream(resources=df, uri=uri)
        assert Path(uri).exists(), "Stream file should exist when URI is a string path"

    @pytest.mark.parametrize("extension", [".mp4", ".gif"])
    def test_stream_with_different_extensions(self, df, extension):
        result = stream(resources=df, extension=extension)
        expected_class = Mp4Stream if extension == ".mp4" else GifStream
        assert isinstance(
            result, expected_class
        ), f"Expected an instance of {expected_class.__name__}"


class TestConnect:
    def test_connect_streams(self):
        stream1 = stream(resources=[0, 1, 2])
        stream2 = stream(resources=[3, 4, 5])
        result = connect(streams=[stream1, stream2])
        assert isinstance(
            result, ConnectedStreams
        ), "Expected an instance of ConnectedStreams"

    def test_connect_streams_with_uri(self, df, tmp_path):
        stream1 = stream(resources=df)
        stream2 = stream(resources=df)
        uri = tmp_path / "connected.mp4"
        connect(streams=[stream1, stream2], uri=uri)
        assert uri.exists(), "Connected stream file should exist"
