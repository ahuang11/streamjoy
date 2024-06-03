from __future__ import annotations

import base64
import gc
from abc import abstractmethod
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from functools import partial
from io import BytesIO
from itertools import zip_longest
from pathlib import Path
from textwrap import indent
from typing import TYPE_CHECKING, Any, Callable

import dask.delayed
import imageio.v3 as iio
import numpy as np
import param
from dask.diagnostics import ProgressBar
from dask.distributed import Client, Future, fire_and_forget
from imageio.core.v3_plugin_api import PluginV3
from PIL import Image, ImageDraw

from . import _utils
from .models import ImageText, Paused
from .serializers import (
    serialize_appropriately,
    serialize_holoviews,
    serialize_pandas,
    serialize_paths,
    serialize_polars,
    serialize_url,
    serialize_xarray,
)
from .settings import config, extension_handlers

if TYPE_CHECKING:
    try:
        import pandas as pd
    except ImportError:
        pd = None

    try:
        import polars as pl
    except ImportError:
        pl = None

    try:
        import xarray as xr
    except ImportError:
        xr = None

    try:
        import holoviews as hv
    except ImportError:
        hv = None

    try:
        import panel as pn
    except ImportError:
        pn = None


class MediaStream(param.Parameterized):
    """
    An abstract class for creating media streams from various resources.
    Do not use this class directly; use either GifStream or Mp4Stream.

    Expand Source code to see all the parameters and descriptions.
    """

    resources = param.List(
        default=None,
        doc="The resources to render.",
        precedence=0
    )

    renderer = param.Callable(
        doc="The renderer to use for the resources.",
    )

    renderer_iterables = param.ClassSelector(
        default=[],
        class_=Sequence,
        doc="Additional positional arguments to pass to the renderer.",
    )

    renderer_kwargs = param.Dict(
        default={},
        doc="Additional keyword arguments to pass to the renderer.",
    )

    write_kwargs = param.Dict(
        default={},
        doc="Additional keyword arguments to pass to the write method.",
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

    ending_pause = param.Number(
        doc="The duration in seconds to pause at the end of the GIF."
    )

    max_frames = param.Integer(
        doc="The maximum number of frames to render.", bounds=(-1, None)
    )

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

    show_progress = param.Boolean(
        default=True,
        doc="Whether to show the progress bar when rendering.",
    )

    scratch_dir = param.Path(
        doc="The directory to use for temporary files.", check_exists=False
    )

    in_memory = param.Boolean(
        doc="Whether to store intermediate results in memory.",
    )

    fsspec_fs = param.Parameter(
        doc="The fsspec filesystem to use for reading and writing.",
    )

    display = param.Boolean(
        doc="Whether to display the output in the notebook after rendering.",
    )

    _tbd_kwargs = param.Dict(doc="Params that are currently unknown.")

    _extension = ""

    logger = _utils.update_logger()

    def __init__(self, resources: list[Any] | None = None, **params) -> None:
        params["resources"] = resources
        params = _utils.populate_config_defaults(params, self.param)

        # forward non recognized params to _tbd_kwargs
        params["_tbd_kwargs"] = {}
        for param_key in set(params):
            if param_key not in self.param:
                params["_tbd_kwargs"][param_key] = params.pop(param_key)

        super().__init__(**params)
        self._progress_bar = ProgressBar(minimum=3 if self.show_progress else np.inf)

    @classmethod
    def from_numpy(
        cls,
        array: np.ndarray,
        renderer: Callable | None = None,
        renderer_iterables: list[Any] | None = None,
        renderer_kwargs: dict | None = None,
        **kwargs,
    ) -> MediaStream:
        serialized = serialize_appropriately(
            cls,
            resources=array,
            renderer=renderer,
            renderer_iterables=renderer_iterables,
            renderer_kwargs=renderer_kwargs,
            **kwargs,
        )
        return cls(**serialized.param.values(), **serialized.kwargs)

    @classmethod
    def from_xarray(
        cls,
        ds: xr.Dataset | xr.DataArray,
        renderer: Callable | None = None,
        renderer_iterables: list[Any] | None = None,
        renderer_kwargs: dict | None = None,
        dim: str | None = None,
        var: str | None = None,
        **kwargs,
    ) -> MediaStream:
        serialized = serialize_xarray(
            cls,
            ds,
            renderer=renderer,
            renderer_iterables=renderer_iterables,
            renderer_kwargs=renderer_kwargs,
            dim=dim,
            var=var,
            **kwargs,
        )
        return cls(**serialized.param.values(), **serialized.kwargs)

    @classmethod
    def from_pandas(
        cls,
        df: pd.DataFrame,
        renderer: Callable | None = None,
        renderer_iterables: list[Any] | None = None,
        renderer_kwargs: dict | None = None,
        groupby: str | None = None,
        **kwargs,
    ) -> MediaStream:
        serialized = serialize_pandas(
            cls,
            resources=df,
            renderer=renderer,
            renderer_iterables=renderer_iterables,
            renderer_kwargs=renderer_kwargs,
            groupby=groupby,
            **kwargs,
        )
        return cls(**serialized.param.values(), **serialized.kwargs)

    @classmethod
    def from_polars(
        cls,
        df: pl.DataFrame,
        renderer: Callable | None = None,
        renderer_iterables: list[Any] | None = None,
        renderer_kwargs: dict | None = None,
        groupby: str | None = None,
        **kwargs,
    ) -> MediaStream:
        serialized = serialize_polars(
            cls,
            resources=df,
            renderer=renderer,
            renderer_iterables=renderer_iterables,
            renderer_kwargs=renderer_kwargs,
            groupby=groupby,
            **kwargs,
        )
        return cls(**serialized.param.values(), **serialized.kwargs)

    @classmethod
    def from_holoviews(
        cls,
        hv_obj: hv.HoloMap | hv.DynamicMap,
        renderer: Callable | None = None,
        renderer_iterables: list[Any] | None = None,
        renderer_kwargs: dict | None = None,
        **kwargs,
    ) -> MediaStream:
        serialized = serialize_holoviews(
            cls,
            resources=hv_obj,
            renderer=renderer,
            renderer_iterables=renderer_iterables,
            renderer_kwargs=renderer_kwargs,
            **kwargs,
        )
        return cls(**serialized.param.values(), **serialized.kwargs)

    @classmethod
    def from_url(
        cls,
        base_url: str,
        pattern: str | None = None,
        sort_key: Callable | None = None,
        max_files: int | None = None,
        file_handler: Callable | None = None,
        file_handler_kwargs: dict | None = None,
        renderer: Callable | None = None,
        renderer_iterables: list[Any] | None = None,
        renderer_kwargs: dict | None = None,
        **kwargs,
    ) -> MediaStream:
        serialized = serialize_url(
            cls,
            resources=base_url,
            pattern=pattern,
            sort_key=sort_key,
            max_files=max_files,
            file_handler=file_handler,
            file_handler_kwargs=file_handler_kwargs,
            renderer=renderer,
            renderer_iterables=renderer_iterables,
            renderer_kwargs=renderer_kwargs,
            **kwargs,
        )
        return cls(**serialized.param.values(), **serialized.kwargs)

    @classmethod
    def from_directory(
        cls,
        base_dir: str | Path,
        pattern: str,
        sort_key: Callable | None = None,
        max_files: int | None = None,
        file_handler: Callable | None = None,
        file_handler_kwargs: dict | None = None,
        renderer: Callable | None = None,
        renderer_iterables: list[Any] | None = None,
        renderer_kwargs: dict | None = None,
        **kwargs,
    ) -> MediaStream:
        paths = Path(base_dir).glob(pattern)
        serialized = serialize_paths(
            cls,
            resources=paths,
            sort_key=sort_key,
            max_files=max_files,
            file_handler=file_handler,
            file_handler_kwargs=file_handler_kwargs,
            renderer=renderer,
            renderer_iterables=renderer_iterables,
            renderer_kwargs=renderer_kwargs,
            **kwargs,
        )
        return cls(**serialized.param.values(), **serialized.kwargs)

    @classmethod
    def _display_in_notebook(
        cls, obj: Any, display: bool = True, is_media: bool = True
    ) -> None:
        if not _utils.using_notebook() or not display:
            return
        from IPython.display import display as ipydisplay

        ipydisplay(obj)

    def _validate_uri(self, path: str | Path, match_extension: bool = True) -> Path:
        if path is None:
            path = _utils.get_config_default("uri", path)

        if isinstance(path, (str, Path)):
            path = Path(path)
            if not path.suffix:
                path = path.with_suffix(self._extension)
            elif path.suffix != self._extension and match_extension:
                raise ValueError(
                    f"Expected {self._extension!r} extension; got {path.suffix!r}."
                )
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _open_buffer(
        self, uri: str | Path | BytesIO, mode: str, extension: str | None = None
    ) -> PluginV3:
        return iio.imopen(uri, mode, extension=extension)

    @abstractmethod
    def _write_images(
        self, buf: PluginV3, images: list[Future], **write_kwargs
    ) -> None:
        """
        This method is responsible for writing images to the output buffer.
        """

    def _render_images(
        self,
        resources: list[Any],
        renderer: Callable | None = None,
        renderer_iterables: list[Any] | None = None,
        renderer_kwargs: dict | None = None,
    ) -> list[Future]:
        try:
            resource_0 = _utils.get_result(_utils.get_first(resources))
        except Exception as exc:
            raise ValueError(
                f"The resources must be set in the class or passed to write; "
                f"got {resources=!r}."
            ) from exc

        if renderer and not renderer.__name__.startswith("default"):
            try:
                iterable_0 = [iterable[0] for iterable in renderer_iterables]
                renderer(resource_0, *iterable_0, **renderer_kwargs)
            except Exception as exc:
                raise exc

        batch_size = self.batch_size
        if renderer is None and "://" in str(resource_0):
            renderer = partial(
                _utils.download_file,
                scratch_dir=self.scratch_dir,
                in_memory=self.in_memory,
            )

        if renderer and self.processes:
            resources = _utils.map_over(
                self.client,
                renderer,
                resources,
                batch_size,
                *renderer_iterables,
                **renderer_kwargs,
            )
        elif renderer and not self.processes:
            renderer = dask.delayed(renderer)
            jobs = [
                renderer(resource, *iterable, **renderer_kwargs)
                for resource, *iterable in zip_longest(resources, *renderer_iterables)
            ]
            with self._progress_bar:
                resources = dask.compute(jobs, scheduler="threads")[0]
        resource_0 = _utils.get_result(_utils.get_first(resources))

        is_like_image = isinstance(resource_0, np.ndarray) and resource_0.ndim == 3
        if not is_like_image:
            try:
                _utils.imread_with_pause(resource_0, fsspec_fs=self.fsspec_fs)
            except Exception as exc:
                raise ValueError(
                    f"Could not read the first resource as an image: {resource_0!r}; "
                    f"if matplotlib or holoviews, try wrapping it with "
                    f"`streamjoy.wrap_matplotlib` or `streamjoy.wrap_holoviews`."
                ) from exc
            images = _utils.map_over(
                self.client,
                _utils.imread_with_pause,
                resources,
                batch_size,
                fsspec_fs=self.fsspec_fs,
            )
        else:
            images = resources

        del resource_0
        return images

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

    def copy(self) -> MediaStream:
        """
        Return a copy of the MediaStream.
        """
        return self.__class__(**self.param.values())

    def write(
        self,
        uri: str | Path | BytesIO | None = None,
        resources: Any | None = None,
        **kwargs: dict[str, Any],
    ) -> Path | BytesIO:
        """
        Write the stream to a file or buffer.

        Args:
            uri (str, Path, BytesIO, optional): The file path or buffer to write to.
            resources (Any, optional): The resources to write.

        Returns:
            Path or BytesIO: The file path or buffer written to.
        """
        uri = self._validate_uri(uri or BytesIO())

        renderer = self.renderer
        renderer_kwargs = self.renderer_kwargs.copy()

        for key, value in self._tbd_kwargs.items():
            if key in self.param:
                setattr(self, key, value)
            else:
                renderer_kwargs[key] = value

        if resources is None:
            resources = self.resources
            renderer_iterables = self.renderer_iterables
            max_frames = _utils.get_max_frames(
                len(resources), kwargs.get("max_frames", self.max_frames)
            )
            kwargs["max_frames"] = max_frames
            resources, renderer_iterables = _utils.subset_resources_renderer_iterables(
                resources, renderer_iterables, max_frames
            )
        else:
            serialized = serialize_appropriately(
                self, resources, renderer, renderer_kwargs, **kwargs
            )
            resources = serialized.resources
            renderer = serialized.renderer
            renderer_iterables = serialized.renderer_iterables
            renderer_kwargs = serialized.renderer_kwargs
            kwargs = serialized.kwargs

        # Do this after for efficiency; also it's possible processes is set to False
        self.client = _utils.get_distributed_client(
            self.client,
            processes=self.processes,
            threads_per_worker=self.threads_per_worker,
        )
        self._display_in_notebook(self.client, display=self.display, is_media=False)

        images = self._render_images(
            resources, renderer, renderer_iterables, renderer_kwargs
        )
        outdated = isinstance(uri, Path) and uri.exists()
        with self._open_buffer(uri, "w", extension=self._extension) as buf:
            self._write_images(buf, images, **self.write_kwargs)
        del images
        del resources
        fire_and_forget(gc.collect)

        green = config["logging_success_color"]
        red = config["logging_warning_color"]
        reset = config["logging_reset_color"]
        if isinstance(uri, Path):
            self.logger.success(f"Saved stream to {green}{uri.absolute()}{reset}.")
        else:
            self.logger.success(f"Saved stream to {green}memory{reset}.")
        if _utils.using_notebook() and self.display and outdated:
            self.logger.warning(
                f"The output displayed below{red} could be an older, cached version{reset}; "
                f"click the path above to view the latest."
            )
        return uri

    def join(self, other: MediaStream) -> MediaStream:
        """
        Connects two homogeneous streams into one by simply extending the resources.
        For heterogeneous streams, use `streamjoy.connect` instead.

        Args:
            other (MediaStream): The other stream with identical params to join.

        Returns:
            MediaStream: The joined stream.
        """
        stream = self.copy()
        if isinstance(other, MediaStream):
            stream.resources = self.resources + other.resources
        return stream

    def __add__(self, other: MediaStream) -> MediaStream:
        """
        Connects two homogeneous streams into one by simply extending the resources.
        """
        return self.join(other)

    def __radd__(self, other: MediaStream) -> MediaStream:
        """
        Connects two homogeneous streams into one by simply extending the resources.
        """
        return self.join(other)

    def __copy__(self) -> MediaStream:
        return self.copy()

    def __len__(self) -> int:
        """
        Return the number of resources.
        """
        return len(self.resources)

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
            f"  fsspec_fs: {self.fsspec_fs}\n"
        )

        if self.intro_title or self.intro_subtitle or self.intro_watermark:
            repr_str += "---\nIntro:\n"
            repr_str += f"  intro_title: {self.intro_title}\n"
            repr_str += f"  intro_subtitle: {self.intro_subtitle}\n"
            repr_str += f"  intro_watermark: {self.intro_watermark}\n"
            repr_str += f"  intro_pause: {self.intro_pause}\n"
            repr_str += f"  intro_background: {self.intro_background}\n"

        client_str = (
            str(self.client).lstrip("<").rstrip(">") if self.client else "Client:"
        )
        repr_str += (
            f"---\n"
            f"{client_str}\n"
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

        if self._tbd_kwargs:
            repr_str += "---\nTBD Keywords (on write):\n"
            for key, value in self._tbd_kwargs.items():
                if isinstance(value, (list, tuple)):
                    value = f"[{', '.join(map(str, value))}]"
                elif isinstance(value, dict):
                    value = f"{{{', '.join(f'{k}: {str(v)[:88]}' for k, v in value.items())}}}"
                repr_str += f"  {key}: {value}\n"

        if self.resources:
            repr_str += (
                f"---\n"
                f"Resources: ({len(self.resources)} frames to stream)\n"
                f"{indent(str(self.resources[0]), ' ' * 2)}\n"
                f"  ...\n"
                f"{indent(str(self.resources[-1]), ' ' * 2)}\n"
            )
        repr_str += "---"
        return repr_str


class Mp4Stream(MediaStream):
    """
    A stream that writes to an MP4.

    See `MediaStream` for all the available parameters.
    """

    from imageio.plugins.pyav import PyAVPlugin

    codec = param.String(doc="The codec to use for the video.")

    _extension = ".mp4"

    def __init__(self, **params) -> None:
        params["codec"] = _utils.get_config_default(
            "codec", params.get("codec"), warn=False
        )
        super().__init__(**params)

    @classmethod
    def _display_in_notebook(
        cls,
        obj: Any,
        display: bool = True,
        is_media: bool = True,
    ) -> None:
        if not _utils.using_notebook() or not display:
            return

        from IPython.display import HTML, Video
        from IPython.display import display as ipydisplay

        if is_media:
            if isinstance(obj, BytesIO):
                obj.seek(0)
                data = base64.b64encode(obj.getvalue()).decode("ascii")
                obj = HTML(
                    data=f"""
                    <video controls>
                        <source src="data:video/mp4;base64,{data}" type="video/mp4">
                    </video>
                """
                )
            else:
                obj = Video(obj)
        ipydisplay(obj)

    def _prepend_intro(self, buf: PyAVPlugin, intro_frame: np.ndarray, **kwargs):
        if intro_frame is None:
            return

        _utils.repeat_frame(
            buf.write_frame, intro_frame, self.intro_pause, self.fps, **kwargs
        )

    def _write_images(
        self, buf: PyAVPlugin, images: list[Future], **write_kwargs
    ) -> None:
        """
        Write images for MP4Stream.
        """
        init = write_kwargs.pop("init", True)
        init_kwargs = _utils.pop_kwargs(buf.init_video_stream, write_kwargs)
        if init:
            init_kwargs["fps"] = self.fps
            buf.init_video_stream(self.codec, **init_kwargs)

        if "crf" in write_kwargs:
            buf._video_stream.options = {"crf": str(write_kwargs.pop("crf"))}

        intro_frame = self._create_intro(images)
        self._prepend_intro(buf, intro_frame, **write_kwargs)

        for i, image in enumerate(images):
            image = _utils.get_result(image)

            pause = None
            if isinstance(image, Paused):
                pause = image.seconds
                image = image.output

            if image.shape[0] % 2:
                image = image[:-1, :, :]
            if image.shape[1] % 2:
                image = image[:, :-1, :]
            image = image[:, :, :3]  # remove alpha channel if present
            buf.write_frame(image, **write_kwargs)

            if pause is not None:
                _utils.repeat_frame(
                    buf.write_frame, image, pause, self.fps, **write_kwargs
                )

            if i == len(images) - 1:
                _utils.repeat_frame(
                    buf.write_frame, image, self.ending_pause, self.fps, **write_kwargs
                )
            del image

    def write(
        self,
        uri: str | Path | BytesIO | None = None,
        resources: Any | None = None,
        **kwargs: dict[str, Any],
    ) -> Path | BytesIO:
        """
        Write the MP4 stream to a file or in memory.

        Args:
            uri: The file path or BytesIO object to write to.
            resources: The resources to write to the file or in memory.
            **kwargs: Additional keyword arguments to pass.

        Returns:
            The file path or BytesIO object.
        """
        uri = self._validate_uri(uri)
        uri = super().write(uri=uri, resources=resources, **kwargs)
        self._display_in_notebook(uri, display=self.display)
        return uri


class GifStream(MediaStream):
    """
    A stream that writes to a gif.

    See `MediaStream` for all the available parameters.
    """

    from imageio.plugins.pillow import PillowPlugin

    loop = param.Integer(doc="The number of times to loop the gif; 0 means infinite.")

    optimize = param.Boolean(
        doc="Whether to optimize the gif with pygifsicle; may take longer to render."
    )

    _extension = ".gif"

    def __init__(self, **params) -> None:
        params["loop"] = _utils.get_config_default(
            "loop", params.get("loop"), warn=False
        )
        super().__init__(**params)

    @classmethod
    def _display_in_notebook(
        cls, obj: Any, display: bool = True, is_media: bool = True
    ) -> None:
        if not _utils.using_notebook() or not display:
            return

        from IPython.display import HTML, Image
        from IPython.display import display as ipydisplay

        if is_media:
            if isinstance(obj, BytesIO):
                obj.seek(0)
                data = base64.b64encode(obj.getvalue()).decode("ascii")
                obj = HTML(
                    data=f"""
                        <img src="data:image/gif;base64,{data}">
                    """
                )
            else:
                obj = Image(obj)
        ipydisplay(obj)

    def _prepend_intro(self, buf: PillowPlugin, intro_frame: np.ndarray, **kwargs):
        if intro_frame is None:
            return

        _utils.repeat_frame(
            buf.write, intro_frame, self.intro_pause, self.fps, **kwargs
        )

    def _compute_duration(
        self, intro_title: str, intro_subtitle: str, num_frames: int
    ) -> float | list[float]:
        if intro_title or intro_subtitle:
            num_frames += int(self.fps * self.intro_pause)

        duration = np.repeat(1 / self.fps, num_frames)
        duration[-1] = self.ending_pause
        duration = (duration * 1000).tolist()
        return duration

    def _write_images(
        self, buf: PillowPlugin, images: list[Future], **write_kwargs
    ) -> None:
        if "duration" in write_kwargs:
            duration = write_kwargs.pop("duration")
        else:
            num_frames = len(images)
            duration = self._compute_duration(
                self.intro_title, self.intro_subtitle, num_frames
            )
        intro_frame = self._create_intro(images)

        if len(duration) == 1:
            duration = duration[0]
        write_kwargs = self.write_kwargs.copy()
        write_kwargs.update(loop=self.loop, is_batch=False, duration=duration)

        self._prepend_intro(buf, intro_frame, **write_kwargs)

        for i, image in enumerate(images):
            image = _utils.get_result(image)
            if isinstance(image, Paused):
                duration[i] = image.seconds * 1000
                image = image.output
            buf.write(image[:, :, :3], **write_kwargs)
            del image

    def _optimize_gif(self, uri: Path) -> None:
        try:
            import pygifsicle
        except ImportError:
            raise ImportError(
                "The `pygifsicle` package is required to optimize gifs; "
                "install it with `pip install pygifsicle`."
            )

        options = [
            "--optimize=2",
            "--loopcount=0",
            "--no-warnings",
            "--no-conserve-memory",
        ]
        optimized_path = uri.with_stem(f"{uri.stem}_optimized")
        pygifsicle.gifsicle(
            sources=uri,
            destination=optimized_path,
            options=options,
        )
        if not optimized_path.stat().st_size:
            raise RuntimeError(
                "gifsicle failed somewhere; ensure that the inputs, "
                "`items`, `uri`, `gifsicle_options` are valid"
            )

        optimized_path.rename(uri)
        green = config["logging_success_color"]
        reset = config["logging_reset_color"]
        self.logger.success(
            f"Optimized stream with pygifsicle at " f"{green}{uri.absolute()}{reset}."
        )

    def write(
        self,
        uri: str | Path | BytesIO | None = None,
        resources: Any | None = None,
        **kwargs: dict[str, Any],
    ) -> Path | BytesIO:
        """
        Write the gif stream to a file or in memory.

        Args:
            uri: The file path or BytesIO object to write to.
            resources: The resources to write to the file or in memory.
            **kwargs: Additional keyword arguments to pass.

        Returns:
            The file path or BytesIO object.
        """
        optimize = kwargs.pop("optimize", self.optimize)
        uri = self._validate_uri(uri)
        uri = super().write(uri=uri, resources=resources, **kwargs)
        self._display_in_notebook(uri, display=self.display)
        if optimize:
            self._optimize_gif(uri)
        return uri


class AnyStream(MediaStream):
    """
    A stream that can be materialized into an Mp4Stream or GifStream.

    See `MediaStream` for all the available parameters.
    """

    @classmethod
    def _display_in_notebook(
        cls, obj: Any, display: bool = True, is_media: bool = True
    ) -> None:
        return

    def materialize(
        self, uri: Path | BytesIO, extension: str | None = None
    ) -> MediaStream:
        """
        Materialize the stream into an Mp4Stream, GifStream, or HtmlStream.
        """
        if isinstance(uri, BytesIO) and extension is None:
            extension = ".mp4"
        elif extension is None:
            extension = uri.suffix

        if isinstance(extension, str) and extension not in extension_handlers:
            raise ValueError(f"Unsupported extension: {extension}")

        stream_cls_str = extension_handlers.get(extension)
        if stream_cls_str is None:
            raise ValueError(
                f"Unsupported extension: {extension}; select from {extension_handlers}"
            )
        stream_cls = globals()[stream_cls_str]
        stream = stream_cls(**self.param.values())
        return stream

    def write(
        self,
        uri: str | Path | BytesIO | None = None,
        resources: Any | None = None,
        extension: str | None = None,
        **kwargs: dict[str, Any],
    ) -> Path | BytesIO:
        """
        Write the stream to a file or in memory.

        Args:
            uri: The file path or BytesIO object to write to.
            resources: The resources to write to the file or in memory.
            extension: The extension to use; useful if uri is a file-like object.
            **kwargs: Additional keyword arguments to pass.
        """
        uri = self._validate_uri(uri or BytesIO(), match_extension=False)
        stream = self.materialize(uri, extension)
        return stream.write(uri=uri, resources=resources, **kwargs)


class HtmlStream(MediaStream):
    _extension = ".html"

    width = param.Integer(default=None, bounds=(1, None), doc="The width of the image.")
    height = param.Integer(
        default=None, bounds=(1, None), doc="The height of the image."
    )
    sizing_mode = param.Selector(
        default="scale_width",
        objects=[
            "fixed",
            "stretch_width",
            "stretch_height",
            "stretch_both",
            "scale_width",
            "scale_height",
            "scale_both",
        ],
        doc="The sizing mode of the image.",
    )

    def __init__(self, **params) -> None:
        import panel as pn

        pn.extension()
        super().__init__(**params)
        self._column = pn.Column()

    @contextmanager
    def _open_buffer(
        self, uri: str | Path | BytesIO, mode: str, extension: str | None = None
    ):
        import panel as pn

        tabs = pn.Tabs(
            tabs_location="right",
            stylesheets=[
                """
                .bk-header {
                    opacity: 0.2; /* Initially hide the element */
                    transition: opacity 0.5s ease; /* Smooth transition for the opacity change */
                }

                .bk-header:hover {
                    opacity: 1; /* Make the element fully visible on hover */
                }
                """
            ],  # noqa: E501
        )
        player = pn.widgets.Player(
            name="Time",
            start=0,
            value=0,
            loop_policy="loop",
            interval=int(1000 / self.fps),
            stylesheets=[
                """
                :host(.bk-panel-models-widgets-Player) {
                    opacity: 0.2;
                    transition: opacity 0.5s ease;
                }

                :host(.bk-panel-models-widgets-Player:hover) {
                    opacity: 1;
                }
                """
            ],
        )
        player.jslink(tabs, value="active", bidirectional=True)
        self._column.objects = [tabs, player]
        yield tabs
        image = tabs.objects[0]
        width = image.object.width
        height = image.object.height
        with param.parameterized.batch_call_watchers(self):
            if self.sizing_mode == "fixed":
                tabs.param.update(
                    width=width + 50,
                    height=height,
                )
                player.param.update(
                    width=width,
                    end=len(tabs) - 1,
                )
                self._column.param.update(
                    width=width,
                    height=height + 100,
                )
            else:
                sizing_mode = self.sizing_mode.replace("both", "width")
                tabs.param.update(
                    min_height=300,
                    max_height=int(height * 1.5),
                    sizing_mode=sizing_mode,
                )
                player.param.update(
                    max_height=150,
                    max_width=450,
                    sizing_mode=sizing_mode,
                    end=len(tabs) - 1,
                    stylesheets=[
                        """
                        :host {
                            align-self: center;
                        }
                        """
                    ],
                )
                self._column.param.update(
                    min_height=300,
                    max_height=int(height * 1.5),
                )
        self._column.save(uri)

    def _write_images(self, buf: pn.Tabs, images: list[Future], **write_kwargs) -> None:
        import panel as pn
        from PIL import Image

        intro_frame = self._create_intro(images)
        self._prepend_intro(buf, intro_frame, **write_kwargs)

        for i, image in enumerate(images):
            image = Image.fromarray(_utils.get_result(image))

            pause = None
            if isinstance(image, Paused):
                pause = image.seconds
                image = image.output

            sizing_mode = self.sizing_mode
            image_tuple = (
                str(i),
                pn.pane.Image(
                    image,
                    width=(
                        (self.width or image.width) if sizing_mode == "fixed" else None
                    ),
                    height=(
                        (self.height or image.height)
                        if sizing_mode == "fixed"
                        else None
                    ),
                    sizing_mode=self.sizing_mode,
                ),
            )
            buf.append(image_tuple, **write_kwargs)

            if pause is not None:
                _utils.repeat_frame(
                    buf.append, image_tuple, pause, self.fps, **write_kwargs
                )

            if i == len(images) - 1:
                _utils.repeat_frame(
                    buf.append,
                    image_tuple,
                    self.ending_pause,
                    self.fps,
                    **write_kwargs,
                )
            del image

    def write(
        self,
        uri: str | Path | BytesIO | None = None,
        resources: Any | None = None,
        **kwargs: dict[str, Any],
    ) -> Path | BytesIO:
        """
        Write the player stream to a file or in memory.

        Args:
            uri: The file path or BytesIO object to write to.
            resources: The resources to write to the file or in memory.
            **kwargs: Additional keyword arguments to pass.

        Returns:
            The file path or BytesIO object.
        """
        self._display_in_notebook(self._column, display=self.display)
        uri = self._validate_uri(uri)
        uri = super().write(uri=uri, resources=resources, **kwargs)
        return uri if not isinstance(uri, BytesIO) else self._column


class ConnectedStreams(param.Parameterized):
    """
    Multiple streams connected together.
    """

    streams = param.List(
        default=[],
        item_type=MediaStream,
        doc="The streams to connect.",
    )

    def __init__(self, streams: list[MediaStream] | None = None, **params) -> None:
        params["streams"] = streams or params.get("streams")
        self.logger = _utils.update_logger()
        super().__init__(**params)

    def write(
        self,
        uri: str | Path | BytesIO | None = None,
        extension: str | None = None,
        **kwargs: dict[str, Any],
    ) -> Path | BytesIO:
        """
        Write the connected streams to a file or in memory.

        Args:
            uri: The file path or BytesIO object to write to.
            extension: The file extension to use.
            **kwargs: Additional keyword arguments to pass.

        Returns:
            The file path or BytesIO object.
        """
        if uri is None:
            uri = BytesIO()
        elif isinstance(uri, str):
            uri = Path(uri)
        streams = [
            (
                stream.materialize(uri, extension)
                if isinstance(stream, AnyStream)
                else stream
            )
            for stream in self.streams
        ]

        client = _utils.get_distributed_client(
            client=kwargs.get("client"),
            processes=kwargs.get("processes"),
            threads_per_worker=kwargs.get("threads_per_worker"),
        )

        stream_0 = streams[0]
        extension = stream_0._extension

        duration = []
        if isinstance(stream_0, GifStream):
            for stream in streams:
                duration += stream._compute_duration(
                    stream.intro_title, stream.intro_subtitle, len(stream)
                )

        with iio.imopen(uri, "w", extension=extension) as buf:
            for i, stream in enumerate(streams):
                stream.client = client
                resources, renderer_iterables = (
                    _utils.subset_resources_renderer_iterables(
                        stream.resources, stream.renderer_iterables, stream.max_frames
                    )
                )
                images = stream._render_images(
                    resources,
                    stream.renderer,
                    renderer_iterables,
                    stream.renderer_kwargs,
                )
                write_kwargs = stream.write_kwargs.copy()
                if isinstance(stream, GifStream):
                    write_kwargs["duration"] = duration
                else:
                    write_kwargs["init"] = i == 0
                stream._write_images(buf, images, **write_kwargs)
                if isinstance(stream, GifStream) and stream.optimize:
                    stream._optimize_gif(uri)

        green = config["logging_success_color"]
        reset = config["logging_reset_color"]
        if isinstance(uri, Path):
            self.logger.success(f"Saved stream to {green}{uri.absolute()}{reset}.")
        else:
            self.logger.success(f"Saved stream to {green}memory{reset}.")
        stream._display_in_notebook(uri, display=stream.display)
        return uri

    def __add__(self, other: MediaStream) -> ConnectedStreams:
        """
        Add a stream to the existing list of streams.
        """
        if isinstance(other, MediaStream):
            self.streams.append(other)
        return self

    def __radd__(self, other: MediaStream) -> ConnectedStreams:
        """
        Add a stream to the existing list of streams.
        """
        return self.__add__(other)

    def __repr__(self) -> str:
        repr_str = (
            f"<{self.__class__.__name__}>\n"
            f"---\n"
            f"Streams:\n"
            f"{indent(repr(self.streams[0]), ' ' * 2)}\n"
            f"  ...\n"
            f"{indent(repr(self.streams[-1]), ' ' * 2)}\n"
        )
        return repr_str

    def __len__(self) -> int:
        """
        Return the total number of frames in all streams.
        """
        return sum(len(stream) for stream in self.streams)

    def __iter__(self) -> Iterable[Path]:
        """
        Iterate over the streams.
        """
        return self.streams
