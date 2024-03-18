from __future__ import annotations

import base64
import gc
from abc import abstractmethod
from collections.abc import Iterable, Sequence
from functools import partial
from inspect import isgenerator
from io import BytesIO
from itertools import zip_longest
from pathlib import Path
from textwrap import indent
from typing import TYPE_CHECKING, Any, Callable

import dask.delayed
import imageio.v3 as iio
import numpy as np
import param
from dask.distributed import Client, Future, fire_and_forget
from imageio.core.v3_plugin_api import PluginV3
from PIL import Image, ImageDraw

from . import _utils
from .models import ImageText, Paused
from .renderers import (
    default_holoviews_renderer,
    default_pandas_renderer,
    default_xarray_renderer,
)
from .settings import config, file_handlers, obj_handlers
from .wrappers import wrap_holoviews, wrap_matplotlib

if TYPE_CHECKING:
    try:
        import pandas as pd
    except ImportError:
        pd = None

    try:
        import xarray as xr
    except ImportError:
        xr = None

    try:
        import holoviews as hv
    except ImportError:
        hv = None


class MediaStream(param.Parameterized):
    """
    An abstract class for creating media streams from various resources.
    Do not use this class directly; use either GifStream or Mp4Stream.

    Expand Source code to see all the parameters and descriptions.
    """

    resources = param.List(
        default=None,
        doc="The resources to render.",
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

    scratch_dir = param.Path(
        doc="The directory to use for temporary files.", check_exists=False
    )

    in_memory = param.Boolean(
        doc="Whether to store intermediate results in memory.",
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

    @classmethod
    def _select_obj_handler(cls, resources: Any) -> MediaStream:
        if isinstance(resources, str) and "://" in resources:
            return cls._expand_from_url
        if isinstance(resources, (Path, str)):
            return cls._expand_from_paths

        for class_or_package_name, method_name in obj_handlers.items():
            module = getattr(resources, "__module__", "").split(".", maxsplit=1)[0]
            type_ = type(resources).__name__
            if (
                f"{module}.{type_}" == class_or_package_name
                or module == class_or_package_name
            ):
                return getattr(cls, method_name)

        raise ValueError(
            f"Could not find a method to handle {type(resources)}; "
            f"supported classes/packages are {list(obj_handlers.keys())}."
        )

    @classmethod
    def _expand_from_xarray(
        cls,
        resources: xr.Dataset | xr.DataArray,
        renderer: Callable | None = None,
        renderer_iterables: list[Any] | None = None,
        renderer_kwargs: dict | None = None,
        **kwargs,
    ):
        ds = resources
        dim = kwargs.pop("dim", None)
        var = kwargs.pop("var", None)

        ds = _utils.validate_xarray(ds, dim=dim, var=var)
        if not dim:
            dim = list(ds.dims)[0]
            _utils.warn_default_used("dim", dim, suffix="from the dataset")
        elif dim not in ds.dims:
            raise ValueError(f"{dim!r} not in {ds.dims!r}")

        total_frames = len(ds[dim])
        max_frames = _utils.get_max_frames(total_frames, kwargs.get("max_frames"))
        resources = [ds.isel({dim: i}) for i in range(max_frames)]

        renderer_kwargs = renderer_kwargs or {}
        renderer_kwargs.update(_utils.pop_from_cls(cls, kwargs))

        if renderer is None:
            renderer = wrap_matplotlib(
                in_memory=kwargs.get("in_memory"),
                scratch_dir=kwargs.get("scratch_dir"),
            )(default_xarray_renderer)
            ds_0 = resources[0]
            if ds_0.ndim >= 2:
                renderer_kwargs["vmin"] = renderer_kwargs.get(
                    "vmin", _utils.get_result(ds_0.min()).item()
                )
                renderer_kwargs["vmax"] = renderer_kwargs.get(
                    "vmax", _utils.get_result(ds_0.max()).item()
                )
            else:
                renderer_kwargs["ylim"] = renderer_kwargs.get(
                    "ylim",
                    (
                        _utils.get_result(ds_0.min()).item(),
                        _utils.get_result(ds_0.max()).item(),
                    ),
                )
        return resources, renderer, renderer_iterables, renderer_kwargs, kwargs

    @classmethod
    def _expand_from_pandas(
        cls,
        resources: pd.DataFrame,
        renderer: Callable | None = None,
        renderer_iterables: list[Any] | None = None,
        renderer_kwargs: dict | None = None,
        **kwargs,
    ):
        import pandas as pd

        df = resources
        groupby = kwargs.get("groupby")

        total_frames = df.groupby(groupby).size().max() if groupby else len(df)
        max_frames = _utils.get_max_frames(total_frames, kwargs.get("max_frames"))
        resources = [
            df.groupby(groupby, as_index=False).head(i) if groupby else df.head(i)
            for i in range(1, max_frames + 1)
        ]

        renderer_kwargs = renderer_kwargs or {}
        renderer_kwargs.update(_utils.pop_from_cls(cls, kwargs))

        if renderer is None:
            renderer = wrap_matplotlib(
                in_memory=kwargs.get("in_memory"),
                scratch_dir=kwargs.get("scratch_dir"),
            )(default_pandas_renderer)
            if "x" not in renderer_kwargs:
                if df.index.name or isinstance(df, pd.Series):
                    renderer_kwargs["x"] = df.index.name
                else:
                    for col in df.columns:
                        if col != groupby:
                            break
                    renderer_kwargs["x"] = col
                _utils.warn_default_used(
                    "x", renderer_kwargs["x"], suffix="from the dataframe"
                )
            if "y" not in renderer_kwargs:
                if isinstance(df, pd.Series):
                    col = df.name
                else:
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

        return resources, renderer, renderer_iterables, renderer_kwargs, kwargs

    @classmethod
    def _expand_from_holoviews(
        cls,
        resources: hv.HoloMap | hv.DynamicMap,
        renderer: Callable | None = None,
        renderer_iterables: list[Any] | None = None,
        renderer_kwargs: dict | None = None,
        **kwargs,
    ):
        import holoviews as hv

        backend = hv.Store.current_backend
        hv.extension(backend)

        def _select_element(hv_obj, key):
            try:
                resource = hv_obj[key]
            except Exception:
                resource = hv_obj.select(**{kdims[0].name: key})
            return resource

        hv_obj = resources
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
        renderer_kwargs.update(_utils.pop_from_cls(cls, kwargs))

        if renderer is None:
            renderer = wrap_holoviews(
                in_memory=kwargs.get("in_memory"),
                scratch_dir=kwargs.get("scratch_dir"),
            )(default_holoviews_renderer)
            clims = {}
            for hv_el in hv_obj.traverse(full_breadth=False):
                if isinstance(hv_el, hv.DynamicMap):
                    hv.render(hv_el, backend=backend)

                if isinstance(hv_el, hv.Element):
                    if hv_el.ndims > 1:
                        vdim = hv_el.vdims[0].name
                        array = hv_el.dimension_values(vdim)
                        clim = (np.nanmin(array), np.nanmax(array))
                        clims[vdim] = clim
            renderer_kwargs.update(
                backend=backend,
                clims=clims,
            )

        if kwargs.get("processes"):
            cls.logger.warning(
                "HoloViews rendering does not support processes; "
                "setting processes=False."
            )
        kwargs["processes"] = False
        return resources, renderer, renderer_iterables, renderer_kwargs, kwargs

    @classmethod
    def _expand_from_url(
        cls,
        resources: str,
        renderer: Callable | None = None,
        renderer_iterables: list[Any] | None = None,
        renderer_kwargs: dict | None = None,
        **kwargs,
    ) -> MediaStream:
        import re

        import requests
        from bs4 import BeautifulSoup

        base_url = resources
        pattern = kwargs.pop("pattern", None)
        sort_key = kwargs.pop("sort_key", None)
        max_files = kwargs.pop("max_files", None)
        file_handler = kwargs.pop("file_handler", None)
        file_handler_kwargs = kwargs.pop("file_handler_kwargs", None)

        response = requests.get(resources)
        content_type = response.headers.get("Content-Type")

        if pattern is not None:
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            href = re.compile(pattern.replace("*", ".*"))
            links = soup.find_all("a", href=href)
        else:
            if content_type.startswith("text"):
                raise ValueError(
                    f"A pattern must be provided if the URL is a directory of files; "
                    f"got {resources!r}."
                )
            links = [{"href": ""}]

        max_files = _utils.get_config_default(
            "max_files", max_files, total_value=len(links), suffix="links"
        )
        if max_files > 0:
            links = links[:max_files]

        if len(links) == 0:
            raise ValueError(
                f"No links found with pattern {pattern!r} at {base_url!r}."
            )

        # download files
        urls = [base_url + link.get("href") for link in links]
        client = _utils.get_distributed_client(
            client=kwargs.get("client"),
            processes=kwargs.get("processes"),
            threads_per_worker=kwargs.get("threads_per_worker"),
        )

        futures = _utils.map_over(
            client,
            _utils.download_file,
            urls,
            kwargs.get("batch_size"),
            scratch_dir=kwargs.get("scratch_dir"),
            in_memory=kwargs.get("in_memory"),
        )
        paths = client.gather(futures)
        return cls._expand_from_paths(
            paths,
            sort_key=sort_key,
            max_files=max_files,
            file_handler=file_handler,
            file_handler_kwargs=file_handler_kwargs,
            renderer=renderer,
            renderer_iterables=renderer_iterables,
            renderer_kwargs=renderer_kwargs,
            **kwargs,
        )

    @classmethod
    def _expand_from_paths(
        cls,
        resources: list[str | Path] | str,
        renderer: Callable | None = None,
        renderer_iterables: list[Any] | None = None,
        renderer_kwargs: dict | None = None,
        **kwargs,
    ) -> MediaStream:
        if isinstance(resources, str):
            resources = [resources]

        sort_key = kwargs.pop("sort_key", None)
        max_files = kwargs.pop("max_files", None)
        file_handler = kwargs.pop("file_handler", None)
        file_handler_kwargs = kwargs.pop("file_handler_kwargs", None)

        paths = sorted(resources, key=sort_key)

        max_files = _utils.get_config_default(
            "max_files", max_files, total_value=len(paths), suffix="paths"
        )
        if max_files > 0:
            paths = paths[:max_files]

        # find a file handler
        extension = Path(paths[0]).suffix
        file_handler_meta = file_handlers.get(extension, {})
        file_handler_import_path = file_handler_meta.get("import_path")
        file_handler_concat_path = file_handler_meta.get("concat_path")
        if file_handler is None and file_handler_import_path is not None:
            file_handler = _utils.import_function(file_handler_import_path)
            if file_handler_concat_path is not None:
                file_handler_concat = _utils.import_function(file_handler_concat_path)

        # read as objects
        if file_handler is not None:
            if file_handler_concat_path is not None:
                resources = file_handler_concat(
                    file_handler(path, **(file_handler_kwargs or {})) for path in paths
                )
            else:
                resources = file_handler(paths, **(file_handler_kwargs or {}))
            return cls._expand_from_any(
                resources, renderer, renderer_iterables, renderer_kwargs, **kwargs
            )

        # or simply return image paths
        return paths, renderer, renderer_iterables, renderer_kwargs, kwargs

    @classmethod
    def _subset_resources_renderer_iterables(
        cls, resources: Any, renderer_iterables: list[Any], max_frames: int
    ):
        if len(resources) > max_frames and max_frames != -1:
            color = config["logging_warning_color"]
            reset = config["logging_reset_color"]
            cls.logger.warning(
                f"There are a total of {len(resources)} frames, "
                f"but streaming only {color}{max_frames}{reset}. "
                f"Set max_frames to -1 to stream all frames."
            )
        resources = resources[: max_frames or max_frames]
        renderer_iterables = [
            iterable[: len(resources)] for iterable in renderer_iterables or []
        ]
        return resources, renderer_iterables

    @classmethod
    def _expand_from_any(
        cls,
        resources: Any,
        renderer: Callable,
        renderer_iterables: list[Any],
        renderer_kwargs: dict[str, Any],
        **kwargs,
    ):
        if not (isinstance(resources, (list, tuple)) or isgenerator(resources)):
            obj_handler = cls._select_obj_handler(resources)
            _utils.validate_renderer_iterables(resources, renderer_iterables)

            resources, renderer, renderer_iterables, renderer_kwargs, kwargs = (
                obj_handler(
                    resources,
                    renderer=renderer,
                    renderer_iterables=renderer_iterables,
                    renderer_kwargs=renderer_kwargs,
                    **kwargs,
                )
            )
            max_frames = _utils.get_max_frames(len(resources), kwargs.get("max_frames"))
            kwargs["max_frames"] = max_frames
            resources, renderer_iterables = cls._subset_resources_renderer_iterables(
                resources, renderer_iterables, max_frames
            )
            _utils.pop_kwargs(obj_handler, renderer_kwargs)
            _utils.pop_kwargs(obj_handler, kwargs)
        return resources, renderer, renderer_iterables, renderer_kwargs, kwargs

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
        resources, renderer, renderer_iterables, renderer_kwargs, kwargs = (
            cls._expand_from_xarray(
                ds,
                renderer=renderer,
                renderer_kwargs=renderer_kwargs,
                dim=dim,
                var=var,
                **kwargs,
            )
        )
        return cls(
            resources=resources,
            renderer=renderer,
            renderer_iterables=renderer_iterables,
            renderer_kwargs=renderer_kwargs,
            **kwargs,
        )

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
        resources, renderer, renderer_iterables, renderer_kwargs, kwargs = (
            cls._expand_from_pandas(
                df, renderer, renderer_kwargs, groupby=groupby, **kwargs
            )
        )
        return cls(
            resources=resources,
            renderer=renderer,
            renderer_iterables=renderer_iterables,
            renderer_kwargs=renderer_kwargs,
            **kwargs,
        )

    @classmethod
    def from_holoviews(
        cls,
        hv_obj: hv.HoloMap | hv.DynamicMap,
        renderer: Callable | None = None,
        renderer_iterables: list[Any] | None = None,
        renderer_kwargs: dict | None = None,
        **kwargs,
    ) -> MediaStream:
        resources, renderer, renderer_iterables, renderer_kwargs, kwargs = (
            cls._expand_from_holoviews(hv_obj, renderer, renderer_kwargs, **kwargs)
        )
        return cls(
            resources=resources,
            renderer=renderer,
            renderer_iterables=renderer_iterables,
            renderer_kwargs=renderer_kwargs,
            **kwargs,
        )

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
        resources, renderer, renderer_iterables, renderer_kwargs, kwargs = (
            cls._expand_from_url(
                base_url,
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
        )
        return cls(
            resources=resources,
            renderer=renderer,
            renderer_iterables=renderer_iterables,
            renderer_kwargs=renderer_kwargs,
            **kwargs,
        )

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
        resources, renderer, renderer_iterables, renderer_kwargs, kwargs = (
            cls._expand_from_paths(
                paths,
                sort_key=sort_key,
                max_files=max_files,
                file_handler=file_handler,
                file_handler_kwargs=file_handler_kwargs,
                renderer=renderer,
                renderer_iterables=renderer_iterables,
                renderer_kwargs=renderer_kwargs,
                **kwargs,
            )
        )
        return cls(
            resources=resources,
            renderer=renderer,
            renderer_iterables=renderer_iterables,
            renderer_kwargs=renderer_kwargs,
            **kwargs,
        )

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

        if renderer:
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
            resources = dask.compute(jobs, scheduler="threads")[0]
        resource_0 = _utils.get_result(_utils.get_first(resources))

        is_like_image = isinstance(resource_0, np.ndarray) and resource_0.ndim == 3
        if not is_like_image:
            try:
                _utils.imread_with_pause(resource_0)
            except Exception as exc:
                raise ValueError(
                    f"Could not read the first resource as an image: {resource_0!r}; "
                    f"if matplotlib or holoviews, try wrapping it with "
                    f"`streamjoy.wrap_matplotlib` or `streamjoy.wrap_holoviews`."
                ) from exc
            images = _utils.map_over(
                self.client, _utils.imread_with_pause, resources, batch_size
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
        else:
            resources, renderer, renderer_iterables, renderer_kwargs, kwargs = (
                self._expand_from_any(resources, renderer, renderer_kwargs, **kwargs)
            )

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
        with iio.imopen(uri, "w", extension=self._extension) as buf:
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

        intro_frame = self._create_intro(images)
        self._prepend_intro(buf, intro_frame, **write_kwargs)

        for image in images:
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

        _utils.repeat_frame(
            buf.write_frame, image, self.ending_pause, self.fps, **write_kwargs
        )

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
        params["ending_pause"] = _utils.get_config_default(
            "ending_pause", params.get("ending_pause"), warn=False
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

        for image in images:
            image = _utils.get_result(image)
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
    ) -> Mp4Stream | GifStream:
        """
        Materialize the stream into an Mp4Stream or GifStream.
        """
        if isinstance(uri, BytesIO) and extension is None:
            extension = ".mp4"
        elif extension is None:
            extension = uri.suffix

        if extension == ".mp4":
            stream_cls = Mp4Stream
        elif extension == ".gif":
            stream_cls = GifStream
        else:
            raise ValueError(
                f"Unsupported file extension {extension!r}; "
                "expected '.mp4' or '.gif'."
            )
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
                    stream._subset_resources_renderer_iterables(
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
