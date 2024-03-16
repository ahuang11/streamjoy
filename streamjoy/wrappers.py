from __future__ import annotations

from functools import wraps
from io import BytesIO
from pathlib import Path

from . import _utils
from .settings import config
from .models import Paused


def wrap_matplotlib(in_memory: bool = False, scratch_dir: str | Path | None = None):
    def wrapper(renderer):
        @wraps(renderer)
        def wrapped(*args, **kwargs) -> Path | BytesIO:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.rcParams.update({"figure.max_open_warning": config["max_open_warning"]})

            output = renderer(*args, **kwargs)

            fig = output
            return_paused = False
            if isinstance(output, Paused):
                return_paused = True
                fig = output.output

            if isinstance(fig, plt.Axes):
                fig = fig.figure
            elif not isinstance(fig, plt.Figure):
                raise ValueError("Renderer must return a Figure or Axes object.")

            uri = _utils.resolve_uri(
                file_name=f"{hash(fig)}.jpg",
                scratch_dir=scratch_dir,
                in_memory=in_memory,
            )
            fig.savefig(uri, format="jpg")
            plt.close(fig)
            return (
                uri if not return_paused else Paused(output=uri, seconds=output.seconds)
            )

        return wrapped

    return wrapper


def wrap_holoviews(in_memory: bool = False, scratch_dir: str | Path | None = None):
    def wrapper(renderer):
        @wraps(renderer)
        def wrapped(*args, **kwargs) -> Path | BytesIO:
            import holoviews as hv

            backend = kwargs.get("backend", hv.Store.current_backend)
            hv.extension(backend)

            output = renderer(*args, **kwargs)

            hv_obj = output
            return_paused = False
            if isinstance(output, Paused):
                return_paused = True
                hv_obj = output.output

            uri = _utils.resolve_uri(
                file_name=f"{hash(hv_obj)}.png",
                scratch_dir=scratch_dir,
                in_memory=in_memory,
            )
            if backend == "bokeh":
                from bokeh.io.export import export_png
                from selenium.webdriver.chrome.options import Options
                from selenium.webdriver.chrome.webdriver import Service, WebDriver
                from webdriver_manager.chrome import ChromeDriverManager

                options = Options()
                options.add_argument("--headless")
                options.add_argument("--disable-extensions")
                with WebDriver(
                    service=Service(ChromeDriverManager().install()), options=options
                ) as webdriver:
                    export_png(
                        hv.render(hv_obj, backend="bokeh"),
                        filename=uri,
                        webdriver=webdriver,
                    )
            else:
                hv.save(hv_obj, uri, fmt="png")

            return (
                uri if not return_paused else Paused(output=uri, seconds=output.seconds)
            )

        return wrapped

    return wrapper
