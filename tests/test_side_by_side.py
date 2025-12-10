from io import BytesIO
from pathlib import Path

import pytest
from imageio.v3 import improps

from streamjoy.core import side_by_side, stream
from streamjoy.streams import SideBySideStreams


class TestSideBySide:
    def test_side_by_side_no_uri(self, df):
        stream1 = stream(resources=df)
        stream2 = stream(resources=df)
        result = side_by_side(streams=[stream1, stream2])
        assert isinstance(
            result, SideBySideStreams
        ), "Expected an instance of SideBySideStreams"

    def test_side_by_side_with_uri(self, df, tmp_path):
        stream1 = stream(resources=df)
        stream2 = stream(resources=df)
        uri = tmp_path / "side_by_side.mp4"
        side_by_side(streams=[stream1, stream2], uri=uri)
        assert uri.exists(), "Side-by-side stream file should exist"

    def test_side_by_side_with_bytesio(self, df):
        stream1 = stream(resources=df)
        stream2 = stream(resources=df)
        result = side_by_side(streams=[stream1, stream2], uri=BytesIO())
        assert isinstance(
            result, BytesIO
        ), "Expected a BytesIO object when URI is a BytesIO instance"

    def test_side_by_side_gif(self, df, tmp_path):
        stream1 = stream(resources=df, extension=".gif")
        stream2 = stream(resources=df, extension=".gif")
        uri = tmp_path / "side_by_side.gif"
        side_by_side(streams=[stream1, stream2], uri=uri)
        assert uri.exists(), "Side-by-side GIF file should exist"

        # Verify output properties
        props = improps(uri)
        assert props.n_images == 3, "Expected 3 frames in the output"

    def test_side_by_side_different_frame_counts(self, df, tmp_path):
        # Create streams with different frame counts
        stream1 = stream(resources=df, max_frames=2, extension=".gif")
        stream2 = stream(resources=df, max_frames=3, extension=".gif")
        uri = tmp_path / "side_by_side_diff.gif"
        side_by_side(streams=[stream1, stream2], uri=uri)
        assert uri.exists(), "Side-by-side stream file should exist"

        # Should use the maximum frame count
        props = improps(uri)
        assert props.n_images == 3, "Expected 3 frames (max of both streams)"

    def test_side_by_side_multiple_streams(self, df, tmp_path):
        # Test with more than 2 streams
        stream1 = stream(resources=df, extension=".gif")
        stream2 = stream(resources=df, extension=".gif")
        stream3 = stream(resources=df, extension=".gif")
        uri = tmp_path / "side_by_side_three.gif"
        side_by_side(streams=[stream1, stream2, stream3], uri=uri)
        assert uri.exists(), "Side-by-side stream file with 3 streams should exist"

        props = improps(uri)
        assert props.n_images == 3, "Expected 3 frames in the output"

    def test_side_by_side_len(self, df):
        stream1 = stream(resources=df, max_frames=2)
        stream2 = stream(resources=df, max_frames=3)
        sbs = side_by_side(streams=[stream1, stream2])
        assert len(sbs) == 3, "Length should be the maximum frame count"

    def test_side_by_side_repr(self, df):
        stream1 = stream(resources=df)
        stream2 = stream(resources=df)
        sbs = side_by_side(streams=[stream1, stream2])
        repr_str = repr(sbs)
        assert "SideBySideStreams" in repr_str
        assert "side by side" in repr_str.lower()
