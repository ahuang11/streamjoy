from __future__ import annotations

import gc
from abc import abstractmethod
from textwrap import indent
from itertools import zip_longest
from pathlib import Path
from typing import Any

import dask.delayed
import imageio.v3 as iio
import numpy as np
import param
from dask.distributed import Client, Future, fire_and_forget
from imageio.core.v3_plugin_api import PluginV3

from . import _utils
from .settings import config, readers
from .wrappers import wrap_holoviews, wrap_matplotlib


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

    max_frames = param.Integer(
        default=None, doc="The maximum number of frames to render."
    )

    fps = param.Integer(
        default=15,
        doc="The number of frames per second to use for the output.",
    )

    renderer = param.Callable(
        doc="A callable that renders the resources and outputs a uri or image."
    )

    renderer_kwargs = param.Dict(
        default={},
        doc="A dictionary of keyword arguments to be passed to the renderer.",
    )

    batch_size = param.Integer(
        default=None,
        doc="The number of resources to process in a single batch.",
    )

    processes = param.Boolean(
        default=True,
        doc="Whether to allow both processes and threads to be used.",
    )

    threads_per_worker = param.Integer(
        default=None,
        doc="The number of threads to use per worker.",
    )

    client = param.ClassSelector(
        class_=Client,
        default=None,
        doc="""
        The distributed client to use; if None, tries to use an existing client,
        or creates a new one.
        """,
    )

    display = param.Boolean(
        default=True,
        doc="Whether to display the output in the notebook after rendering.",
    )

    def __init__(self, resources: list, **params) -> None:
        self.logger = _utils.update_logger()
        params["resources"] = resources
        if hasattr(resources, "tolist"):
            params["resources"] = params["resources"].tolist()
        params["fps"] = _utils.get_config_default("fps", params.get("fps"), warn=False)
        params["batch_size"] = _utils.get_config_default(
            "batch_size", params.get("batch_size"), warn=False
        )
        super().__init__(**params)
        self.client = _utils.get_distributed_client(
            self.client,
            processes=self.processes,
            threads_per_worker=self.threads_per_worker,
        )
        self._display_in_notebook(self.client)
        self._extension = ""

    def _display_in_notebook(self, obj: Any) -> None:
        if not _utils.using_notebook() or not self.display:
            return
        from IPython.display import display

        display(obj)

    @classmethod
    def _select_method(cls, resources: Any):
        if isinstance(resources, list):
            return cls
        elif isinstance(resources, str) and "://" in resources:
            return cls.from_url

        for class_name, method_name in readers.items():
            # module of resources
            module = getattr(resources, "__module__", None).split(".", maxsplit=1)[0]
            type_ = type(resources).__name__
            if f"{module}.{type_}" == class_name:
                return getattr(cls, method_name)

        raise ValueError(
            f"Could not find a method to handle {type(resources)}; "
            f"supported types are {list(readers.keys())}."
        )

    @classmethod
    def from_xarray(
        cls,
        ds: "xr.Dataset" | "xr.DataArray",
        dim: str | None = None,
        var: str | None = None,
        **kwargs,
    ) -> _MediaStream:
        import xarray as xr

        @wrap_matplotlib()
        def _default_xarray_renderer(da_sel, **kwargs):
            import matplotlib.pyplot as plt

            fig = plt.figure()
            ax = plt.axes(**kwargs.pop("subplot_kws", {}))
            da_sel.plot(ax=ax, **kwargs)
            return fig

        if var:
            ds = ds[var]
        elif isinstance(ds, xr.Dataset):
            var = list(ds.data_vars)[0]
            _utils.warn_default_used("var", var, suffix="from the dataset")
            ds = ds[var]

        if ds.ndim > 3:
            raise ValueError(f"Can only handle 3D arrays; {ds.ndim}D array found")
        if not dim:
            dim = list(ds.dims)[0]
            _utils.warn_default_used("dim", dim, suffix="from the dataset")
        elif dim not in ds.dims:
            raise ValueError(f"{dim!r} not in {ds.dims!r}")

        max_frames = _utils.get_max_frames(len(ds[dim]), kwargs.get("max_frames", None))
        kwargs["max_frames"] = max_frames
        resources = [ds.isel({dim: i}) for i in range(max_frames)]
        if kwargs.get("renderer") is None:
            kwargs["renderer"] = _default_xarray_renderer

            ds_0 = resources[0]
            renderer_kwargs = kwargs.get("renderer_kwargs", {})
            renderer_kwargs["vmin"] = renderer_kwargs.get(
                "vmin", _utils.get_result(ds_0.min())
            )
            renderer_kwargs["vmax"] = renderer_kwargs.get(
                "vmax", _utils.get_result(ds_0.max())
            )
            renderer_kwargs["extend"] = renderer_kwargs.get("both")
            kwargs["renderer_kwargs"] = renderer_kwargs
        return cls(resources, **kwargs)

    @classmethod
    def from_pandas(
        cls,
        df: "pd.DataFrame",
        groupby: str | None = None,
        **kwargs,
    ) -> _MediaStream:

        @wrap_matplotlib()
        def _default_pandas_renderer(df_sub, **kwargs):
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            for group, df_group in df_sub.groupby(groupby):
                df_group.plot(ax=ax, label=group, **kwargs)
            return fig

        max_frames = _utils.get_max_frames(
            df.groupby(groupby).size().max() if groupby else len(df),
            kwargs.get("max_frames", None),
        )
        resources = [
            df.groupby(groupby, as_index=False).head(i) if groupby else df.head(i)
            for i in range(1, max_frames + 1)
        ]
        if kwargs.get("renderer") is None:
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
        **kwargs,
    ):
        import holoviews as hv

        def _select_element(hv_obj, key):
            try:
                resource = hv_obj[key]
            except Exception:
                resource = hv_obj.select(**{kdims[0].name: key})
            return resource

        @wrap_holoviews()
        def _default_holoviews_renderer(hv_obj, key: Any, **kwargs):
            title = ""
            for hv_el in hv_obj.traverse(full_breadth=False):
                try:
                    vdim = hv_el.vdims[0].name
                except IndexError:
                    continue
                if vdim in clims:
                    hv_el.opts(clim=clims[vdim])
            hv_obj.opts(title=str(key) or title, toolbar=None)
            return hv_obj

        if isinstance(hv_obj, (hv.core.spaces.DynamicMap, hv.core.spaces.HoloMap)):
            hv_map = hv_obj
        elif issubclass(
            type(hv_obj), (hv.core.layout.Layoutable, hv.core.overlay.Overlayable)
        ):
            hv_map = hv_obj[0]

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

        resources = [_select_element(hv_obj, key) for key in keys]
        if kwargs.get("renderer") is None:
            kwargs["renderer"] = _default_holoviews_renderer
            renderer_kwargs = kwargs.get("renderer_kwargs", {}).copy()
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
            kwargs["renderer_kwargs"] = renderer_kwargs

        if include_keys:
            iterables = kwargs.get("iterables", [])
            kwargs["iterables"] = [keys] + iterables

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

        scratch_dir = Path(_utils.get_config_default("scratch_dir", scratch_dir))
        scratch_dir.mkdir(exist_ok=True)

        client = _utils.get_distributed_client(kwargs.pop("client", None))
        response = httpx.get(base_url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        href = re.compile(pattern.replace("*", ".*"))
        links = soup.find_all("a", href=href)

        if url_limit is None:
            url_limit = config["url_limit"]
            _utils.warn_default_used(
                "url_limit", f"{url_limit}", total_value=len(links), suffix="links."
            )
            links = links[:url_limit]
        elif url_limit > 0:
            links = links[:url_limit]

        futures = [
            client.submit(_utils.download_file, base_url, scratch_dir, link.get("href"))
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
            path = _utils.get_config_default("output_path", path)
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

    def copy(self) -> _MediaStream:
        return self.__class__(**self.param.values())

    def write(
        self, output_path: str | Path | None = None, fps: int | None = None, **kwargs
    ) -> None:
        output_path = self._validate_path(output_path)
        fps = _utils.get_config_default("fps", fps, warn=False)

        max_frames = _utils.get_max_frames(len(self.resources), self.max_frames)
        resources = self.resources[:max_frames]
        iterables = self.iterables[:max_frames]
        batch_size = self.batch_size
        resource_0 = _utils.get_result(resources[0])

        if self.renderer:
            try:
                iterable = [] if not iterables else iterables[0]
                self.renderer(resource_0, *iterable, **self.renderer_kwargs)
            except Exception as exc:
                raise exc

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
        resource_0 = _utils.get_result(resources[0])

        is_like_image = isinstance(resource_0, np.ndarray) and resource_0.ndim == 3
        if not is_like_image:
            try:
                iio.imread(resource_0)
            except Exception as exc:
                raise ValueError(
                    f"Expected a valid image URI; got {type(resource_0)}"
                ) from exc
            images = self.client.map(iio.imread, resources, batch_size=batch_size)
        else:
            images = resources

        with iio.imopen(output_path, "w", extension=self._extension) as buf:
            self._write_images(buf, images, **kwargs)
        del images
        del resource_0
        del resources
        fire_and_forget(gc.collect)

        color = config["logging_success_color"]
        reset = config["logging_reset_color"]
        self.logger.success(f"Saved stream to {color}{output_path.absolute()}{reset}.")
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
        frames = len(self.resources)
        repr_str = (
            f"<{self.__class__.__name__}>\n"
            f"---\n"
            f"Output:\n"
            f"  max_frames: {self.max_frames}\n"
            f"  fps: {self.fps}\n"
            f"  display: {self.display}\n"
        )
        repr_str += (
            f"---\n"
            f"{str(self.client).lstrip('<').rstrip('>')}\n"
            f"  batch_size: {self.batch_size}\n"
            f"  processes: {self.processes}\n"
            f"  threads_per_worker: {self.threads_per_worker}\n"
        )
        if self.renderer:
            repr_str += f"---\nRenderer: `{self.renderer.__name__}`\n"
            for key, value in self.renderer_kwargs.items():
                if isinstance(value, (list, tuple)):
                    value = f"[{', '.join(map(str, value))}]"
                elif isinstance(value, dict):
                    value = f"{{{', '.join(f'{k}: {str(v)[:88]}' for k, v in value.items())}}}"
                repr_str += f"  {key}: {value}\n"
        repr_str += (
            f"---\n"
            f"Resources: ({frames} frames)\n"
            f"{indent(str(self.resources[0]), ' ' * 2)}\n"
            f"    ...\n"
            f"{indent(str(self.resources[-1]), ' ' * 4)}\n"
        )
        if self.iterables:
            repr_str += (
                f"---\n"
                f"Iterables: ({len(self.iterables)} iterables)\n"
                f"{indent(str(self.iterables[0]), ' ' * 2)}\n"
                f"    ...\n"
                f"{indent(str(self.iterables[-1]), ' ' * 2)}\n"
            )
        repr_str += f"---\n"
        return repr_str


class Mp4Stream(_MediaStream):
    from imageio.plugins.pyav import PyAVPlugin

    mp4_codec = param.String(default=None, doc="The codec to use for the video.")

    def __init__(self, resources: list, **params) -> None:
        params["mp4_codec"] = _utils.get_config_default(
            "mp4_codec", params.get("mp4_codec"), warn=False
        )
        super().__init__(resources, **params)
        self._extension = ".mp4"

    def _display_in_notebook(
        self,
        obj: Any,
    ) -> None:
        from IPython.display import Video

        try:
            return super()._display_in_notebook(Video(obj))
        except Exception:
            return super()._display_in_notebook(obj)

    def _write_images(self, buf: PyAVPlugin, images: list[Future], **kwargs) -> None:
        buf.init_video_stream(self.mp4_codec, fps=self.fps)
        for image in images:
            image = _utils.get_result(image)
            if image.shape[0] % 2:
                image = image[:-1, :, :]
            if image.shape[1] % 2:
                image = image[:, :-1, :]
            buf.write_frame(image[:, :, :3], **kwargs)

    def write(self, output_path: str | Path | None = None, **kwargs) -> None:
        output_path = self._validate_path(output_path)
        output_path = super().write(output_path, **kwargs)
        self._display_in_notebook(str(output_path))
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
        params["gif_loop"] = _utils.get_config_default(
            "gif_loop", params.get("gif_loop"), warn=False
        )
        params["gif_pause"] = _utils.get_config_default(
            "gif_pause", params.get("gif_pause"), warn=False
        )
        super().__init__(resources, **params)
        self._extension = ".gif"

    def _display_in_notebook(self, obj: Any) -> None:
        from IPython.display import Image

        try:
            return super()._display_in_notebook(Image(obj))
        except Exception:
            return super()._display_in_notebook(obj)

    def _write_images(self, buf: PillowPlugin, images: list[Future], **kwargs) -> None:
        duration = np.repeat(1 / self.fps, len(images))
        duration[-1] = self.gif_pause
        duration = (duration * 1000).tolist()
        kwargs.update(loop=self.gif_loop, is_batch=False, duration=duration)
        for image in images:
            image_3d = _utils.get_result(image)[:, :, :3]
            buf.write(image_3d, **kwargs)
            del image
            del image_3d

    def write(self, output_path: str | Path | None = None, **kwargs) -> None:
        output_path = self._validate_path(output_path)
        output_path = super().write(output_path, **kwargs)
        self._display_in_notebook(output_path)
        return output_path
