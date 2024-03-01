from __future__ import annotations

from functools import wraps
from pathlib import Path

from . import _utils
from .settings import config


def wrap_matplotlib(in_memory: bool = False, scratch_dir: str | Path | None = None):
    def wrapper(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.rcParams.update({"figure.max_open_warning": config["max_open_warning"]})

            fig = func(*args, **kwargs)
            if isinstance(fig, plt.Axes):
                fig = fig.figure
            elif not isinstance(fig, plt.Figure):
                raise ValueError("Function must return a Figure or Axes object.")

            uri = _utils.resolve_uri(in_memory, scratch_dir, f"{id(fig)}.jpg")
            fig.savefig(uri, format="jpg")
            plt.close(fig)
            return uri

        return wrapped

    return wrapper


def wrap_holoviews(in_memory: bool = False, scratch_dir: str | Path | None = None):
    def wrapper(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            import holoviews as hv

            backend = kwargs.get("backend", hv.Store.current_backend)
            hv.extension(backend)

            hv_obj = func(*args, **kwargs)

            uri = _utils.resolve_uri(in_memory, scratch_dir, f"{id(hv_obj)}.png")
            if backend == "bokeh":
                from bokeh.io.export import export_png
                from selenium.webdriver.chrome.options import Options
                from selenium.webdriver.chrome.webdriver import WebDriver
                from webdriver_manager.chrome import ChromeDriverManager

                options = Options()
                options.add_argument("--headless")
                options.add_argument("--disable-extensions")
                with WebDriver(
                    ChromeDriverManager().install(), options=options
                ) as webdriver:
                    export_png(
                        hv.render(hv_obj, backend="bokeh"),
                        filename=uri,
                        webdriver=webdriver,
                    )
            else:
                hv.save(hv_obj, uri, fmt="png")

            return uri

        return wrapped

    return wrapper
