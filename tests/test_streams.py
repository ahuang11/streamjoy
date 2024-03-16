import pytest

class TestMediaStream:

    def test_renderer(self):
        ...

    def test_from_pandas(self):
        ...

    def test_from_xarray(self):
        ...

    def test_from_holoviews(self):
        ...
    
    def test_from_url(self):
        ...
    
    def test_from_directory(self):
        ...

    def test_intro(self):
        ...

class TestGifStream(TestMediaStream):

    def test_paused_renderer(self):
        ...

class TestMp4Stream(TestMediaStream):

    def test_paused_renderer(self):
        ...