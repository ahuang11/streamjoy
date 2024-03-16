import pytest
from streamjoy.wrappers import wrap_matplotlib, wrap_holoviews
from matplotlib import pyplot as plt
from pathlib import Path
from io import BytesIO
from streamjoy.models import Paused
import holoviews as hv


class TestWrapMatplotlib:
    def test_wrap_matplotlib_figure_to_file(self, tmp_path):
        @wrap_matplotlib(in_memory=False, scratch_dir=tmp_path)
        def render_figure():
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [1, 2, 3])
            return fig

        path = render_figure()
        assert Path(path).exists()

    def test_wrap_matplotlib_figure_to_memory(self):
        @wrap_matplotlib(in_memory=True)
        def render_figure():
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [1, 2, 3])
            return fig

        output = render_figure()
        assert isinstance(output, BytesIO)

    def test_wrap_matplotlib_with_paused(self):
        @wrap_matplotlib(in_memory=True)
        def render_figure():
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [1, 2, 3])
            return Paused(output=fig, seconds=5)

        output = render_figure()
        assert isinstance(output, Paused)
        assert isinstance(output.output, BytesIO)
        assert output.seconds == 5


class TestWrapHoloViews:
    def test_wrap_holoviews_to_file(self, tmp_path):
        @wrap_holoviews(in_memory=False, scratch_dir=tmp_path)
        def render_hv():
            curve = hv.Curve((range(10), range(10)))
            return curve

        path = render_hv()
        assert Path(path).exists()

    def test_wrap_holoviews_to_memory(self):
        @wrap_holoviews(in_memory=True)
        def render_hv():
            curve = hv.Curve((range(10), range(10)))
            return curve

        output = render_hv()
        assert isinstance(output, BytesIO)

    def test_wrap_holoviews_with_paused(self):
        @wrap_holoviews(in_memory=True)
        def render_hv():
            curve = hv.Curve((range(10), range(10)))
            return Paused(output=curve, seconds=5)

        output = render_hv()
        assert isinstance(output, Paused)
        assert isinstance(output.output, BytesIO)
        assert output.seconds == 5
