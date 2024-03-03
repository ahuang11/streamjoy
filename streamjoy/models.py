from __future__ import annotations

import gc
import re
from abc import abstractmethod
from itertools import zip_longest
from pathlib import Path
from textwrap import indent
from typing import Any, Callable

import dask.delayed
import imageio.v3 as iio
import numpy as np
import param
from dask.distributed import Client, Future, fire_and_forget
from imageio.core.v3_plugin_api import PluginV3
from PIL import Image, ImageDraw, ImageFont

from . import _utils
from .renderers import (
    default_holoviews_renderer,
    default_pandas_renderer,
    default_xarray_renderer,
)
from .settings import config, file_handlers, obj_handlers
from .wrappers import wrap_holoviews, wrap_matplotlib


class ImageText(param.Parameterized):
    text = param.String(
        doc="The text to render.",
    )

    font = param.String(
        doc="The font to use for the text.",
    )

    size = param.Integer(
        doc="The font size to use for the text.",
    )

    color = param.String(
        doc="The color to use for the text.",
    )

    anchor = param.String(
        doc="The anchor to use for the text.",
    )

    x = param.Integer(
        doc="The x-coordinate to use for the text.",
    )

    y = param.Integer(
        doc="The y-coordinate to use for the text.",
    )

    kwargs = param.Dict(
        default={},
        doc="Additional keyword arguments to pass to the text renderer.",
    )

    def __init__(self, text: str, **params) -> None:
        params["text"] = text
        params = _utils.populate_config_defaults(
            params, self.param.params(), config_prefix="image_text"
        )
        super().__init__(**params)

    def render(
        self,
        draw: ImageDraw,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        x = self.x or width // 2
        y = self.y or height // 2
        try:
            font = ImageFont.truetype(self.font, self.size)
        except Exception:
            font = ImageFont.load_default()
        draw.text(
            (x, y),
            self.text,
            font=font,
            fill=self.color,
            anchor=self.anchor,
            **self.kwargs,
        )


class _MediaStream(param.Parameterized):
    renderer = param.Callable(
        doc="The renderer to use for the resources.",
    )

    renderer_kwargs = param.Dict(
        default={},
        doc="Additional keyword arguments to pass to the renderer.",
    )

    intro_title = param.ClassSelector(
        class_=(str, ImageText),
        doc="""
        The title to use in the intro frame.
        To customize the text, pass an ImageText instance.
        """,
    )

    intro_subtitle = param.ClassSelector(
        class_=(str, ImageText),
        doc="""
        The subtitle to use in the intro frame.
        To customize the text, pass an ImageText instance.
        """,
    )

    intro_watermark = param.ClassSelector(
        class_=(str, ImageText),
        doc="""
        The watermark to use in the intro frame.
        To customize the text, pass an ImageText instance.
        """,
    )

    intro_pause = param.Number(
        default=None,
        bounds=(1, None),
        doc="""
        The duration in seconds to display the intro frame if
        `intro_title` or `intro_subtitle` is set.
        """,
    )

    intro_background = param.Color(
        doc="The background color to use in the intro frame.",
    )

    max_frames = param.Integer(doc="The maximum number of frames to render.")

    fps = param.Integer(
        default=None,
        bounds=(1, None),
        doc="The number of frames per second to use for the output.",
    )

    batch_size = param.Integer(
        default=None,
        bounds=(1, None),
        doc="The number of resources to process in a single batch.",
    )

    client = param.ClassSelector(
        class_=Client,
        doc="""
        The distributed client to use; if None, tries to use an existing client,
        or creates a new one.
        """,
    )

    processes = param.Boolean(
        doc="Whether to allow both processes and threads to be used.",
    )

    threads_per_worker = param.Integer(
        default=None,
        bounds=(1, None),
        doc="The number of threads to use per worker.",
    )

    scratch_dir = param.Path(
        doc="The directory to use for temporary files.", check_exists=False
    )

    in_memory = param.Boolean(
        doc="Whether to store intermediate results in memory.",
    )

    display = param.Boolean(
        doc="Whether to display the output in the notebook after rendering.",
    )

    def __init__(self, **params) -> None:
        self.logger = _utils.update_logger()
        params = _utils.populate_config_defaults(
            params, self.param.params(), warn_on=["max_frames"]
        )
        super().__init__(**params)
        self._extension = ""

    def _display_in_notebook(self, obj: Any, is_media: bool = True) -> None:
        if not _utils.using_notebook() or not self.display:
            return
        from IPython.display import display

        display(obj)

    def _validate_output_path(
        self, path: str | Path, match_extension: bool = True
    ) -> Path:
        if path is None:
            path = _utils.get_config_default("output_path", path)
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(self._extension)
        elif path.suffix != self._extension and match_extension:
            raise ValueError(
                f"Expected {self._extension!r} extension; got {path.suffix!r}."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @abstractmethod
    def _write_images(self, buf: PluginV3, images: list[Future], **kwargs) -> None:
        """
        This method is responsible for writing images to the output buffer.
        """

    def _create_intro(self, images: list[Future]) -> np.ndarray:
        title = self.intro_title
        subtitle = self.intro_subtitle
        watermark = self.intro_watermark
        if not (title or subtitle):
            return

        height, width = _utils.get_result(images[0]).shape[:2]
        intro_frame = Image.new("RGB", (width, height), color=self.intro_background)
        draw = ImageDraw.Draw(intro_frame)

        if isinstance(title, str):
            title_x = width // 2
            title_y = height // 2
            title = ImageText(
                text=title,
                x=title_x,
                y=title_y,
                anchor="ms",
                size=50,
                kwargs=dict(stroke_width=1),
            )
        if title:
            title.render(draw, width, height)

        if isinstance(subtitle, str):
            subtitle_x = width // 2
            subtitle_y = (height // 2) + title.size
            subtitle = ImageText(
                text=subtitle,
                x=subtitle_x,
                y=subtitle_y,
                size=25,
                anchor="ms",
            )
        if subtitle:
            subtitle.render(draw, width, height)

        if isinstance(watermark, str):
            watermark_x = width - 20
            watermark_y = height - 20
            watermark = ImageText(
                text=watermark,
                x=watermark_x,
                y=watermark_y,
                size=15,
                anchor="rb",
            )
        if watermark:
            watermark.render(draw, width, height)

        intro_frame = np.array(intro_frame)
        return intro_frame

    @abstractmethod
    def _prepend_intro(self, buf: PluginV3, intro_frame: np.ndarray, **kwargs) -> None:
        """
        This method is responsible for prepending an intro frame to the output buffer.
        """

    def _select_obj_handler(self, resources: Any) -> _MediaStream:
        if isinstance(resources, str) and "://" in resources:
            return self._expand_from_url

        for class_or_package_name, method_name in obj_handlers.items():
            # module of resources
            module = getattr(resources, "__module__", None).split(".", maxsplit=1)[0]
            type_ = type(resources).__name__
            if (
                f"{module}.{type_}" == class_or_package_name
                or module == class_or_package_name
            ):
                return getattr(self, method_name)

        raise ValueError(
            f"Could not find a method to handle {type(resources)}; "
            f"supported types are {list(obj_handlers.keys())}."
        )

    def _expand_from_xarray(
        self,
        ds: "xr.Dataset" | "xr.DataArray",
        renderer: Callable | None = None,
        renderer_kwargs: dict | None = None,
        dim: str | None = None,
        var: str | None = None,
        **kwargs,
    ) -> _MediaStream:
        ds = _utils.validate_xarray(ds, var=var)
        if not dim:
            dim = list(ds.dims)[0]
            _utils.warn_default_used("dim", dim, suffix="from the dataset")
        elif dim not in ds.dims:
            raise ValueError(f"{dim!r} not in {ds.dims!r}")

        total_frames = len(ds[dim])
        max_frames = _utils.get_max_frames(total_frames, self.max_frames)
        resources = [ds.isel({dim: i}) for i in range(max_frames)]

        renderer_kwargs = renderer_kwargs or {}
        if renderer is None:
            renderer = wrap_matplotlib(
                in_memory=self.in_memory,
                scratch_dir=self.scratch_dir,
            )(default_xarray_renderer)
            ds_0 = resources[0]
            renderer_kwargs["vmin"] = renderer_kwargs.get(
                "vmin", _utils.get_result(ds_0.min())
            )
            renderer_kwargs["vmax"] = renderer_kwargs.get(
                "vmax", _utils.get_result(ds_0.max())
            )
        return resources, renderer, renderer_kwargs

    def _expand_from_pandas(
        self,
        df: "pd.DataFrame",
        renderer: Callable | None = None,
        renderer_kwargs: dict | None = None,
        groupby: str | None = None,
        **kwargs,
    ) -> _MediaStream:
        if "groupby" in renderer_kwargs:
            groupby = renderer_kwargs["groupby"]
        else:
            renderer_kwargs["groupby"] = groupby

        total_frames = df.groupby(groupby).size().max() if groupby else len(df)
        max_frames = _utils.get_max_frames(total_frames, self.max_frames)
        resources = [
            df.groupby(groupby, as_index=False).head(i) if groupby else df.head(i)
            for i in range(1, max_frames + 1)
        ]

        renderer_kwargs = renderer_kwargs or {}
        if renderer is None:
            renderer = wrap_matplotlib(
                in_memory=self.in_memory,
                scratch_dir=self.scratch_dir,
            )(default_pandas_renderer)
            if "x" not in renderer_kwargs:
                if df.index.name:
                    renderer_kwargs["x"] = df.index.name
                else:
                    for col in df.columns:
                        if col != groupby and df[col]:
                            break
                    renderer_kwargs["x"] = col
                _utils.warn_default_used(
                    "x", renderer_kwargs["x"], suffix="from the dataframe"
                )
            if "y" not in renderer_kwargs:
                numeric_cols = df.select_dtypes(include="number").columns
                for col in numeric_cols:
                    if col not in (renderer_kwargs["x"], groupby):
                        break
                renderer_kwargs["y"] = col
                _utils.warn_default_used(
                    "y", renderer_kwargs["y"], suffix="from the dataframe"
                )
            if "xlabel" not in renderer_kwargs:
                renderer_kwargs["xlabel"] = (
                    renderer_kwargs["x"].title().replace("_", " ")
                )
            if "ylabel" not in renderer_kwargs:
                renderer_kwargs["ylabel"] = (
                    renderer_kwargs["y"].title().replace("_", " ")
                )

        return resources, renderer, renderer_kwargs

    def _expand_from_holoviews(
        self,
        hv_obj: "hv.HoloMap" | "hv.DynamicMap",
        renderer: Callable | None = None,
        renderer_kwargs: dict | None = None,
        **kwargs,
    ):
        import holoviews as hv

        def _select_element(hv_obj, key):
            try:
                resource = hv_obj[key]
            except Exception:
                resource = hv_obj.select(**{kdims[0].name: key})
            return resource

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

        resources = [_select_element(hv_obj, key).opts(title=str(key)) for key in keys]

        renderer_kwargs = renderer_kwargs or {}
        if renderer is None:
            renderer = wrap_holoviews(
                in_memory=self.in_memory,
                scratch_dir=self.scratch_dir,
            )(default_holoviews_renderer)
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
            renderer_kwargs["clims"] = clims

        if self.processes:
            self.processes = False
            self.logger.warning(
                "HoloViews rendering does not support processes; "
                "setting processes=False."
            )
        return resources, renderer, renderer_kwargs

    def _expand_from_url(
        self,
        base_url: str,
        pattern: str,
        max_urls: int | None = None,
        file_handler: Callable | None = None,
        file_handler_kwargs: dict | None = None,
        renderer: Callable | None = None,
        renderer_kwargs: dict | None = None,
        **kwargs,
    ) -> _MediaStream:
        import re

        import requests
        from bs4 import BeautifulSoup

        response = requests.get(base_url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        href = re.compile(pattern.replace("*", ".*"))
        links = soup.find_all("a", href=href)

        if max_urls is None:
            max_urls = config["max_urls"]
            _utils.warn_default_used(
                "max_urls", f"{max_urls}", total_value=len(links), suffix="links."
            )
            links = links[:max_urls]
        elif max_urls > 0:
            links = links[:max_urls]

        if len(links) == 0:
            raise ValueError(
                f"No links found with pattern {pattern!r} at {base_url!r}."
            )

        # download files
        urls = [base_url + link.get("href") for link in links]
        client = _utils.get_distributed_client(
            client=self.client,
            processes=self.processes,
            threads_per_worker=self.threads_per_worker,
        )
        futures = client.map(_utils.download_file, urls, in_memory=self.in_memory)
        paths = client.gather(futures)

        # find a file handler
        extension = paths[0].suffix
        file_handler_meta = file_handlers.get(extension, {})
        file_handler_import_path = file_handler_meta.get("import_path")
        if file_handler is None and file_handler_import_path is not None:
            file_handler = _utils.import_function(file_handler_import_path)

        # read as objects
        if file_handler is not None:
            resources = file_handler(paths, **(file_handler_kwargs or {}))
            obj_handler = self._select_obj_handler(resources)
            obj_handler_kwargs = _utils.pop_kwargs(obj_handler, kwargs)
            return obj_handler(
                resources, renderer, renderer_kwargs, **obj_handler_kwargs
            )

        # or simply return image paths
        return paths, renderer, renderer_kwargs

    def copy(self) -> _MediaStream:
        return self.__class__(**self.param.values())

    def write(
        self,
        resources: Any,
        output_path: str | Path | None = None,
        iterables: list[Any] | None = None,
        **kwargs,
    ) -> Path:
        output_path = self._validate_output_path(output_path)

        obj_handler = self._select_obj_handler(resources)
        resources, renderer, renderer_kwargs = obj_handler(
            resources,
            renderer=self.renderer,
            renderer_kwargs=self.renderer_kwargs,
            **kwargs,
        )

        self.client = _utils.get_distributed_client(
            self.client,
            processes=self.processes,
            threads_per_worker=self.threads_per_worker,
        )
        self._display_in_notebook(self.client, is_media=False)

        iterables = iterables or []
        resource_0 = _utils.get_result(resources[0])
        if renderer:
            try:
                iterable = [] if not iterables else iterables[0]
                renderer(resource_0, *iterable, **renderer_kwargs)
            except Exception as exc:
                raise exc

        batch_size = self.batch_size
        if renderer and self.processes:
            resources = self.client.map(
                renderer,
                resources,
                *iterables,
                batch_size=batch_size,
                **renderer_kwargs,
            )
        elif renderer and not self.processes:
            renderer = dask.delayed(renderer)
            jobs = [
                renderer(resource, *iterable, **renderer_kwargs)
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
                    f"Could not read the first resource as an image: {resource_0!r}"
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

    def join(self, other: _MediaStream) -> _MediaStream:
        stream = self.copy()
        for param_name, param_value in self.param.values().items():
            if param_value is None and getattr(other, param_name, None) is not None:
                setattr(stream, param_name, getattr(other, param_name))
        return stream

    def __add__(self, other: _MediaStream) -> _MediaStream:
        return self.join(other)

    def __copy__(self) -> _MediaStream:
        return self.copy()

    def __repr__(self) -> str:
        repr_str = (
            f"<{self.__class__.__name__}>\n"
            f"---\n"
            f"Output:\n"
            f"  max_frames: {self.max_frames}\n"
            f"  fps: {self.fps}\n"
            f"  display: {self.display}\n"
            f"  scratch_dir: {self.scratch_dir}\n"
            f"  in_memory: {self.in_memory}\n"
        )

        if self.intro_title or self.intro_subtitle or self.intro_watermark:
            repr_str += f"---\nIntro:\n"
            repr_str += f"  intro_title: {self.intro_title}\n"
            repr_str += f"  intro_subtitle: {self.intro_subtitle}\n"
            repr_str += f"  intro_watermark: {self.intro_watermark}\n"
            repr_str += f"  intro_pause: {self.intro_pause}\n"
            repr_str += f"  intro_pause: {self.intro_background}\n"

        repr_str += (
            f"---\n"
            f"{str(self.client).lstrip('<').rstrip('>')}\n"
            f"  batch_size: {self.batch_size}\n"
            f"  processes: {self.processes}\n"
            f"  threads_per_worker: {self.threads_per_worker}\n"
        )

        if self.renderer or self.renderer_kwargs:
            repr_str += f"---\nRenderer: `{self.renderer.__name__ if self.renderer else 'N/A'}`\n"
            for key, value in self.renderer_kwargs.items():
                if isinstance(value, (list, tuple)):
                    value = f"[{', '.join(map(str, value))}]"
                elif isinstance(value, dict):
                    value = f"{{{', '.join(f'{k}: {str(v)[:88]}' for k, v in value.items())}}}"
                repr_str += f"  {key}: {value}\n"
        repr_str += f"---\n"
        return repr_str


class Mp4Stream(_MediaStream):
    from imageio.plugins.pyav import PyAVPlugin

    codec = param.String(doc="The codec to use for the video.")

    def __init__(self, **params) -> None:
        params["codec"] = _utils.get_config_default(
            "codec", params.get("codec"), warn=False
        )
        super().__init__(**params)
        self._extension = ".mp4"

    def _display_in_notebook(
        self,
        obj: Any,
        is_media: bool = True,
    ) -> None:
        from IPython.display import Video

        if is_media:
            super()._display_in_notebook(
                "The media could be outdated; please open the file for latest."
            )
            return super()._display_in_notebook(Video(obj))
        else:
            return super()._display_in_notebook(obj)

    def _prepend_intro(self, buf: PyAVPlugin, intro_frame: np.ndarray, **kwargs):
        if intro_frame is None:
            return

        for _ in range(int(self.fps * self.intro_pause)):
            buf.write_frame(intro_frame, **kwargs)

    def _write_images(self, buf: PyAVPlugin, images: list[Future], **kwargs) -> None:
        buf.init_video_stream(self.codec, fps=self.fps)

        write_kwargs = _utils.pop_kwargs(buf.write_frame, kwargs)
        intro_frame = self._create_intro(images)
        self._prepend_intro(buf, intro_frame, **write_kwargs)
        for image in images:
            image = _utils.get_result(image)
            if image.shape[0] % 2:
                image = image[:-1, :, :]
            if image.shape[1] % 2:
                image = image[:, :-1, :]
            buf.write_frame(image[:, :, :3], **write_kwargs)

    def write(
        self,
        resources: Any,
        output_path: str | Path | None = None,
        iterables: list[Any] | None = None,
        **kwargs,
    ) -> Path:
        output_path = self._validate_output_path(output_path)
        output_path = super().write(
            resources,
            output_path=output_path,
            iterables=iterables,
            **kwargs,
        )
        self._display_in_notebook(str(output_path))
        return output_path


class GifStream(_MediaStream):
    from imageio.plugins.pillow import PillowPlugin

    loop = param.Integer(doc="The number of times to loop the gif; 0 means infinite.")

    ending_pause = param.Number(
        doc="The duration in seconds to pause at the end of the GIF."
    )

    def __init__(self, **params) -> None:
        params["loop"] = _utils.get_config_default(
            "loop", params.get("loop"), warn=False
        )
        params["ending_pause"] = _utils.get_config_default(
            "ending_pause", params.get("ending_pause"), warn=False
        )
        super().__init__(**params)
        self._extension = ".gif"

    def _display_in_notebook(self, obj: Any, is_media: bool = True) -> None:
        from IPython.display import Image

        if is_media:
            super()._display_in_notebook(
                "The media could be outdated; please open the file for latest."
            )
            return super()._display_in_notebook(Image(obj))
        else:
            return super()._display_in_notebook(obj)

    def _prepend_intro(self, buf: PillowPlugin, intro_frame: np.ndarray, **kwargs):
        if intro_frame is None:
            return

        for _ in range(int(self.fps * self.intro_pause)):
            buf.write(intro_frame, **kwargs)

    def _write_images(self, buf: PillowPlugin, images: list[Future], **kwargs) -> None:
        intro_frame = self._create_intro(images)
        num_frames = len(images)
        if intro_frame is not None:
            num_frames += int(self.fps * self.intro_pause)
        duration = np.repeat(1 / self.fps, num_frames)
        duration[-1] = self.ending_pause
        duration = (duration * 1000).tolist()
        kwargs.update(loop=self.loop, is_batch=False, duration=duration)

        write_kwargs = _utils.pop_kwargs(buf.write, kwargs)
        self._prepend_intro(buf, intro_frame, **write_kwargs)
        for image in images:
            image_3d = _utils.get_result(image)[:, :, :3]
            buf.write(image_3d, **write_kwargs)
            del image
            del image_3d

    def write(
        self,
        resources: Any,
        output_path: str | Path | None = None,
        iterables: list[Any] | None = None,
        **kwargs,
    ) -> Path:
        output_path = self._validate_output_path(output_path)
        output_path = super().write(
            resources,
            output_path=output_path,
            iterables=iterables,
            **kwargs,
        )
        self._display_in_notebook(output_path)
        return output_path


class AnyStream(_MediaStream):
    def _display_in_notebook(self, obj: Any, is_media: bool = True) -> None:
        return

    def write(
        self,
        resources: Any,
        output_path: str | Path | None = None,
        iterables: list[Any] | None = None,
        **kwargs,
    ) -> Path:
        output_path = self._validate_output_path(output_path, match_extension=False)
        if output_path.suffix == ".mp4":
            stream_cls = Mp4Stream
        elif output_path.suffix == ".gif":
            stream_cls = GifStream
        else:
            raise ValueError(
                f"Unsupported file extension {output_path.suffix}; "
                "expected '.mp4' or '.gif'."
            )
        # TODO: extract kwargs?
        stream = stream_cls(**self.param.values())
        return stream.write(
            resources,
            output_path=output_path,
            iterables=iterables,
            **kwargs,
        )
