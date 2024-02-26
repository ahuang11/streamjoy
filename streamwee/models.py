from __future__ import annotations

import gc
import os
from abc import abstractmethod
from io import BytesIO
from pathlib import Path
from itertools import zip_longest
from typing import Any

import imageio.v3 as iio
import numpy as np
import param
import dask.delayed
from dask.distributed import Client, Future
from dask.diagnostics import ProgressBar
from imageio.core.v3_plugin_api import PluginV3

from .settings import config
from .utils import (
    download_file,
    get_distributed_client,
    get_max_frames,
    get_result,
    using_notebook,
    warn_default_used,
    get_config_default,
    update_logger,
)


class _MediaStream(param.Parameterized):

    resources = param.List(doc="A list of resources to be streamed.")

    iterables = param.List(
        default=[],
        doc="""
        A list of iterables to map alongside the resources; useful for
        rendering resources with additional metadata. Each item in the
        list should be the same length as the resources.
        """,
    )

    renderer = param.Callable(
        doc="A callable that renders the resources and outputs a uri or image."
    )

    renderer_kwargs = param.Dict(
        default={},
        doc="A dictionary of keyword arguments to be passed to the renderer.",
    )

    fps = param.Number(default=None, doc="The frames per second of the video.")

    max_frames = param.Integer(
        default=None, doc="The maximum number of frames to render."
    )

    batch_size = param.Integer(
        default=None,
        doc="The number of resources to process in a single batch.",
    )

    processes = param.Boolean(
        default=True,
        doc="Whether to allow both processes and threads to be used.",
    )

    client = param.ClassSelector(
        class_=Client,
        default=None,
        doc="""
        The distributed client to use; if None, tries to use an existing client,
        or creates a new one.
        """,
    )

    cache = param.Dict(
        default={},
        doc="""
        A cache for storing expensive computations. This is useful for
        avoiding redundant computations when rendering the same resource
        multiple times.
        """,
    )

    display = param.Boolean(
        default=True,
        doc="Whether to display the output in the notebook after rendering.",
    )

    def __init__(self, resources: list, **params) -> None:
        update_logger()
        params["resources"] = resources
        params["fps"] = get_config_default("fps", params.get("fps"), warn=False)
        params["batch_size"] = get_config_default(
            "batch_size", params.get("batch_size"), warn=False
        )
        super().__init__(**params)
        self.client = get_distributed_client(self.client, processes=self.processes)
        self._extension = ""

    @classmethod
    def from_xarray(
        cls,
        ds: "xr.Dataset" | "xr.DataArray",
        dim: str | None = None,
        var: str | None = None,
        **kwargs,
    ) -> _MediaStream:
        import xarray as xr

        def _default_xarray_renderer(da_sel, **kwargs):
            import matplotlib.pyplot as plt

            plt.rcParams.update({"figure.max_open_warning": config["max_open_warning"]})

            buf = BytesIO()
            fig, ax = plt.subplots()
            da_sel.plot(ax=ax, vmin=vmin, vmax=vmax, extend="both")
            fig.savefig(buf, format="jpg")
            plt.close(fig)
            return buf

        if var:
            ds = ds[var]
        elif isinstance(ds, xr.Dataset):
            var = list(ds.data_vars)[0]
            warn_default_used("var", var, "from the dataset.")
            ds = ds[var]

        if ds.ndim > 3:
            raise ValueError(f"Can only handle 3D arrays; {ds.ndim}D array found.")
        if not dim:
            dim = list(ds.dims)[0]
            warn_default_used("dim", dim, "from the dataset.")
        elif dim not in ds.dims:
            raise ValueError(f"{dim!r} not in {ds.dims!r}")

        max_frames = get_max_frames(len(ds[dim]), kwargs.get("max_frames", None))
        resources = [ds.isel({dim: i}) for i in range(max_frames)]
        if "renderer" not in kwargs:
            kwargs["renderer"] = _default_xarray_renderer

            ds_0 = resources[0]
            renderer_kwargs = kwargs.get("renderer_kwargs", {})
            if "vmin" not in renderer_kwargs:
                vmin = ds_0.min()
                if hasattr(vmin, "compute"):
                    vmin = vmin.compute()
                renderer_kwargs["vmin"] = vmin
            if "vmax" not in renderer_kwargs:
                vmax = ds_0.max()
                if hasattr(vmax, "compute"):
                    vmax = vmax.compute()
                renderer_kwargs["vmax"] = vmax
            kwargs["renderer_kwargs"] = renderer_kwargs
        return cls(resources, **kwargs)

    @classmethod
    def from_pandas(
        cls,
        df: "pd.DataFrame",
        groupby: str | None = None,
        **kwargs,
    ) -> _MediaStream:
        def _default_pandas_renderer(df_sub, **kwargs):
            import matplotlib.pyplot as plt

            plt.rcParams.update({"figure.max_open_warning": config["max_open_warning"]})

            buf = BytesIO()
            fig, ax = plt.subplots()
            for group, df_group in df_sub.groupby(groupby):
                df_group.plot(ax=ax, label=group, **kwargs)
            fig.savefig(buf, format="jpg")
            plt.close(fig)
            return buf

        max_frames = get_max_frames(
            df.groupby(groupby).size().max() if groupby else len(df),
            kwargs.get("max_frames", None),
        )
        resources = [
            df.groupby(groupby, as_index=False).head(i) if groupby else df.head(i)
            for i in range(1, max_frames + 1)
        ]
        if "renderer" not in kwargs:
            kwargs["renderer"] = _default_pandas_renderer
            renderer_kwargs = kwargs.get("renderer_kwargs", {})
            if "x" not in renderer_kwargs:
                if df.index.name:
                    renderer_kwargs["x"] = df.index.name
                else:
                    for col in df.columns:
                        if col != groupby:
                            break
                    renderer_kwargs["x"] = col
            if "y" not in renderer_kwargs:
                for col in df.columns:
                    if col not in (renderer_kwargs["x"], groupby):
                        break
                renderer_kwargs["y"] = col
            kwargs["renderer_kwargs"] = renderer_kwargs
        return cls(resources, **kwargs)

    @classmethod
    def from_holoviews(
        cls,
        hv_obj: "hv.HoloMap" | "hv.DynamicMap",
        include_keys: bool = True,
        scratch_dir: str | Path | None = None,
        **kwargs,
    ):
        import holoviews as hv
        from webdriver_manager.chrome import ChromeDriverManager
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.webdriver import WebDriver

        scratch_dir = Path(get_config_default("scratch_dir", scratch_dir))
        scratch_dir.mkdir(exist_ok=True)

        def _default_holoviews_renderer(hv_obj, key: Any | None = None, **kwargs):
            backend = kwargs.get("backend", hv.Store.current_backend)
            hv.extension(backend)

            title = ""
            for hv_el in hv_obj.traverse(full_breadth=False):
                if isinstance(hv_el, hv.Element) and key is None:
                    if kdim_key in hv_el.data:
                        title = str(hv_el.data[kdim_key].values)
                try:
                    vdim = hv_el.vdims[0].name
                except IndexError:
                    continue
                if vdim in clims:
                    hv_el.opts(clim=clims[vdim])
            # hv_obj.opts(title=str(key) or title, toolbar=None)
            path = str(scratch_dir / f"{key}.png")

            if backend == "bokeh":
                from bokeh.io.export import export_png
                options = Options()
                options.add_argument("--headless")
                options.add_argument("--disable-extensions")
                with WebDriver(ChromeDriverManager().install(), options=options) as webdriver:
                    export_png(
                        hv.render(hv_obj, backend="bokeh"),
                        filename=path,
                        webdriver=webdriver,
                    )
            else:
                hv.save(hv_obj, path, fmt="png")
            return path

        if isinstance(hv_obj, (hv.core.spaces.DynamicMap, hv.core.spaces.HoloMap)):
            hv_map = hv_obj
        elif issubclass(
            type(hv_obj), (hv.core.layout.Layoutable, hv.core.overlay.Overlayable)
        ):
            hv_map = hv_obj[0]
            if isinstance(hv_map, hv.DynamicMap):
                hv_obj = hv_obj.decollate()

        if not isinstance(hv_map, (hv.core.spaces.DynamicMap, hv.core.spaces.HoloMap)):
            raise ValueError("Can only handle HoloMap and DynamicMap objects.")
        elif isinstance(hv_map, hv.core.spaces.DynamicMap):
            kdims = hv_map.kdims
            keys = hv_map.kdims[0].values
        else:
            kdims = hv_map.kdims
            keys = hv_map.keys()

        if len(kdims) > 1:
            raise ValueError("Can only handle 1D HoloViews objects.")

        kdim_key = kdims[0].name
        resources = [hv_obj.select(**{kdim_key: key}) for key in sorted(keys)]
        renderer_kwargs = kwargs.get("renderer_kwargs", {}).copy()
        if "renderer" not in kwargs:
            kwargs["renderer"] = _default_holoviews_renderer
            clims = {}
            for hv_el in hv_obj.traverse(full_breadth=False):
                if isinstance(hv_el, hv.DynamicMap):
                    hv.render(hv_el)

                if isinstance(hv_el, hv.Element):
                    if hv_el.ndims > 1:
                        vdim = hv_el.vdims[0].name
                        array = hv_el.dimension_values(vdim)
                        clim = (array.min(), array.max())
                        clims[vdim] = clim

        if include_keys:
            iterables = kwargs.get("iterables", []).copy()
            kwargs["iterables"] = iterables + [keys]

        kwargs["renderer_kwargs"] = renderer_kwargs
        kwargs["processes"] = False  # IMPORTANT!
        return cls(resources, **kwargs)

    @classmethod
    def from_url(
        cls,
        base_url: str,
        pattern: str,
        url_limit: int | None = None,
        scratch_dir: str | Path | None = None,
        **kwargs,
    ) -> _MediaStream:
        import re

        import httpx
        from bs4 import BeautifulSoup

        scratch_dir = Path(get_config_default("scratch_dir", scratch_dir))
        scratch_dir.mkdir(exist_ok=True)

        client = get_distributed_client(kwargs.pop("client", None))
        response = httpx.get(base_url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        href = re.compile(pattern.replace("*", ".*"))
        links = soup.find_all("a", href=href)

        if url_limit is None:
            url_limit = config["url_limit"]
            warn_default_used(
                "url_limit", f"{url_limit}", total_value=len(links), suffix="links."
            )
            links = links[:url_limit]
        elif url_limit > 0:
            links = links[:url_limit]

        futures = [
            client.submit(download_file, base_url, scratch_dir, link.get("href"))
            for link in links
        ]
        paths = client.gather(futures)

        if paths[0].suffix == ".nc":
            import xarray as xr

            ds = xr.open_mfdataset(paths)
            return cls.from_xarray(ds, **kwargs)
        return cls(paths, **kwargs)

    def _validate_path(self, path):
        if path is None:
            path = get_config_default("output_path", path)
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(self._extension)
        elif path.suffix != self._extension:
            raise ValueError(
                f"Expected {self._extension!r} extension; got {path.suffix!r}."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @abstractmethod
    def _write_frames(self, buf: PluginV3, images: list[Future], **kwargs) -> None:
        """
        This method is responsible for writing images to the output buffer.
        """
        raise NotImplementedError

    def copy(self) -> _MediaStream:
        return self.__class__(**self.param.values())

    def write(self, output_path: str | Path | None = None, **kwargs) -> None:
        output_path = self._validate_path(output_path)

        max_frames = get_max_frames(len(self.resources), self.max_frames)
        resources = self.resources[:max_frames]
        iterables = self.iterables[:max_frames]
        batch_size = self.batch_size
        if self.renderer and self.processes:
            resources = self.client.map(
                self.renderer,
                resources,
                *iterables,
                batch_size=batch_size,
                **self.renderer_kwargs,
            )
        elif self.renderer and not self.processes:
            renderer = dask.delayed(self.renderer)
            jobs = [
                renderer(resource, *iterable, **self.renderer_kwargs)
                for resource, *iterable in zip_longest(resources, *iterables)
            ]
            resources = dask.compute(jobs, scheduler="threads")[0]
        resource_0 = get_result(resources[0])

        is_like_image = isinstance(resource_0, np.ndarray) and resource_0.ndim == 3
        if not is_like_image:
            images = self.client.map(iio.imread, resources, batch_size=batch_size)
        else:
            images = resources

        with iio.imopen(output_path, "w", extension=self._extension) as buf:
            self._write_images(buf, images, **kwargs)
        del images
        del resource_0
        del resources
        self.client.submit(gc.collect)
        return output_path

    def merge(self, other: _MediaStream) -> _MediaStream:
        stream = self.copy()
        for param_name, param_value in self.param.values():
            if param_value is None and getattr(other, param_name, None) is not None:
                setattr(stream, param_name, getattr(other, param_name))
        return stream

    def __add__(self, other: _MediaStream) -> _MediaStream:
        return self.merge(other)

    def __copy__(self) -> _MediaStream:
        return self.copy()

    def __repr__(self) -> str:
        total_frames = len(self.resources)
        max_frames = get_max_frames(total_frames, self.max_frames)
        renderer = self.renderer.__name__ if self.renderer else None
        if max_frames != total_frames:
            frames_str = f"{max_frames} / {total_frames}"
        else:
            frames_str = total_frames
        render_str = f" with {renderer!r}" if renderer else " without a renderer"
        return (
            f"Streaming {frames_str} frames{render_str} at {self.fps} FPS; "
            f"batch size of {self.batch_size}.\nThe first and last resource:\n\n"
            f"{self.resources[0]}\n{self.resources[-1]}"
        )


class Mp4Stream(_MediaStream):
    from imageio.plugins.pyav import PyAVPlugin

    mp4_codec = param.String(default=None, doc="The codec to use for the video.")

    def __init__(self, resources: list, **params) -> None:
        params["mp4_codec"] = get_config_default(
            "mp4_codec", params.get("mp4_codec"), warn=False
        )
        super().__init__(resources, **params)
        self._extension = ".mp4"

    def _write_images(self, buf: PyAVPlugin, images: list[Future], **kwargs) -> None:
        buf.init_video_stream(self.mp4_codec, fps=self.fps)
        for image in images:
            image = get_result(image)
            if image.shape[0] % 2:
                image = image[:-1, :, :]
            if image.shape[1] % 2:
                image = image[:, :-1, :]
            buf.write_frame(image[:, :, :3], **kwargs)

    def write(self, output_path: str | Path | None = None, **kwargs) -> None:
        output_path = self._validate_path(output_path)
        output_path = super().write(output_path, **kwargs)
        if using_notebook() and self.display:
            from IPython.display import display, Video

            display(Video(str(output_path)))
        return output_path


class GifStream(_MediaStream):
    from imageio.plugins.pillow import PillowPlugin

    gif_loop = param.Integer(
        default=None, doc="The number of times to loop the gif; 0 means infinite."
    )

    gif_pause = param.Number(
        default=None, doc="The duration in seconds to pause at the end of the GIF."
    )

    def __init__(self, resources: list, **params) -> None:
        params["gif_loop"] = get_config_default(
            "gif_loop", params.get("gif_loop"), warn=False
        )
        params["gif_pause"] = get_config_default(
            "gif_pause", params.get("gif_pause"), warn=False
        )
        super().__init__(resources, **params)
        self._extension = ".gif"

    def _write_images(self, buf: PillowPlugin, images: list[Future], **kwargs) -> None:
        duration = np.repeat(1 / self.fps, len(images))
        duration[-1] = self.gif_pause
        duration = (duration * 1000).tolist()
        kwargs.update(loop=self.gif_loop, is_batch=False, duration=duration)
        for image in images:
            image_3d = get_result(image)[:, :, :3]
            buf.write(image_3d, **kwargs)
            del image
            del image_3d

    def write(self, output_path: str | Path | None = None, **kwargs) -> None:
        output_path = self._validate_path(output_path)
        output_path = super().write(output_path, **kwargs)
        if using_notebook() and self.display:
            from IPython.display import display, Image

            display(Image(str(output_path)))
        return output_path
