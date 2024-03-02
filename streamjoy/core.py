from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from . import _utils
from .models import AnyStream, GifStream, Mp4Stream


def stream(
    resources: Any,
    path: str | Path | None = None,
    renderer: Callable | None = None,
    renderer_kwargs: dict | None = None,
    iterables: list[Any] | None = None,
    job_name: str | None = None,
    write_kwargs: dict | None = None,
    **kwargs,
) -> AnyStream | GifStream | Mp4Stream | Path:
    """
    Create a stream from the given resources.

    Args:
        resources: The resources to create a stream from.
        path: The path to write the stream to. If None, the stream is returned.
        renderer: The renderer to use. If None, the default renderer is used.
        renderer_kwargs: Additional keyword arguments to pass to the renderer.
        iterables: A list of iterables to map alongside the resources; useful for
            rendering resources with additional metadata. Each item in the
            list should be the same length as the resources.
        write_kwargs: Additional keyword arguments to pass to the stream's write method.
        **kwargs: Additional keyword arguments to pass to the stream constructor.

    Returns:
        The stream if path is None, otherwise None.
    """
    stream_cls = AnyStream
    if path:
        path = Path(path)
        if path.suffix == ".mp4":
            stream_cls = Mp4Stream
        elif path.suffix == ".gif":
            stream_cls = GifStream
        else:
            raise ValueError(f"Unsupported file extension {path.suffix}")

    kwargs.update(
        {
            "iterables": iterables or [],
            "renderer": renderer,
            "renderer_kwargs": renderer_kwargs or {},
        }
    )
    cls_method = stream_cls._select_method(resources)
    if cls_method == "from_url":
        if "pattern" not in kwargs:
            raise ValueError(
                "The from_url method requires a pattern set, e.g. `pattern='N_*_v3.0.png'`"
            )
    stream = cls_method(resources, **kwargs)

    if path:
        return stream.write(path, **write_kwargs or {})
    return stream
