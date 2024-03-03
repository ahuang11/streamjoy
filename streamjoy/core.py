from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
from functools import partial

from . import _utils
from .models import AnyStream, GifStream, Mp4Stream


def stream(
    resources: Any,
    output_path: str | Path | None = None,
    renderer: Callable | None = None,
    renderer_kwargs: dict | None = None,
    iterables: list[Any] | None = None,
    **kwargs,
) -> AnyStream | GifStream | Mp4Stream | Path:
    """
    Create a stream from the given resources.

    Args:
        resources: The resources to create a stream from.
        output_path: The path to write the stream to. If None, the stream is returned.
        renderer: The renderer to use. If None, the default renderer is used.
        renderer_kwargs: Additional keyword arguments to pass to the renderer.
        iterables: A list of iterables to map alongside the resources; useful for
            rendering resources with additional metadata. Each item in the
            list should be the same length as the resources.
        **kwargs: Additional keyword arguments to pass to the stream constructor.

    Returns:
        The stream if path is None, otherwise None.
    """
    stream_cls = AnyStream
    if output_path:
        output_path = Path(output_path)
        if output_path.suffix == ".mp4":
            stream_cls = Mp4Stream
        elif output_path.suffix == ".gif":
            stream_cls = GifStream
        else:
            raise ValueError(f"Unsupported file extension {output_path.suffix}")

    params = {
        key: kwargs.pop(key) for key in stream_cls.param.values() if key in kwargs
    }
    stream = stream_cls(
        renderer=renderer, renderer_kwargs=renderer_kwargs or {}, **params
    )

    if output_path:
        return stream.write(
            resources,
            output_path=output_path,
            iterables=iterables,
            **kwargs,
        )
    stream.write = partial(stream.write, resources, **kwargs)
    return stream
