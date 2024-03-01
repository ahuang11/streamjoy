from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from . import _utils
from .models import GifStream, Mp4Stream


def stream(
    resources: Any,
    path: str | Path | None = None,
    renderer: Callable | None = None,
    renderer_kwargs: dict | None = None,
    **kwargs,
) -> None:
    """
    Create a stream from the given resources.

    Args:
        resources: The resources to create a stream from.
        path: The path to write the stream to. If None, the stream is returned.
        renderer: The renderer to use. If None, the default renderer is used.
        renderer_kwargs: Additional keyword arguments to pass to the renderer.
        **kwargs: Additional keyword arguments to pass to the stream constructor.

    Returns:
        The stream if path is None, otherwise None.
    """
    stream_cls = Mp4Stream
    if path:
        path = Path(path)
        if path.suffix == ".mp4":
            stream_cls = Mp4Stream
        elif path.suffix == ".gif":
            stream_cls = GifStream
        else:
            raise ValueError(f"Unsupported file extension {path.suffix}")

    params = {"renderer": renderer, "renderer_kwargs": renderer_kwargs or {}}
    params.update(
        {key: kwargs.pop(key) for key in stream_cls.param.values() if key in kwargs}
    )
    from_method = stream_cls._select_method(resources)
    _utils.extract_kwargs(from_method, params, kwargs)
    stream = from_method(resources, **params)

    if path:
        return stream.write(path, **kwargs)
    return stream
