import pytest
from pathlib import Path
from streamjoy.core import stream
from streamjoy.streams import AnyStream, GifStream, Mp4Stream

class TestStream:
    
    def test_no_uri(self, tmp_path):
        result = stream(resources=[0, 1, 2])
        assert isinstance(result, AnyStream), "Expected an instance of AnyStream or its subclass"

    def test_uri(self, df, tmp_path):
        uri = tmp_path / "test.mp4"
        stream(resources=df, uri=uri)
        assert uri.exists(), "Stream file should exist"

    def test_url(self):
        ...

    def test_propagate_renderer_kwargs(self):
        ...
    
    def test_optimize(self):
        ...


class TestConnect:

    def test_no_uri(self):
        ...

    def test_uri(self):
        ...
