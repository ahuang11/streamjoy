from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .streams import AnyStream, GifStream, Mp4Stream, ConnectedStreams
from . import _utils


def stream(
    resources: Any,
    *iterables,
    output_path: str | Path | None = None,
    renderer: Callable | None = None,
    renderer_kwargs: dict | None = None,
    **kwargs,
) -> AnyStream | GifStream | Mp4Stream | Path:
    """
    Create a stream from the given resources.

    Args:
        resources: The resources to create a stream from.
        output_path: The path to write the stream to. If None, the stream is returned.
        renderer: The renderer to use. If None, the default renderer is used.
        renderer_kwargs: Additional keyword arguments to pass to the renderer.
        **kwargs: Additional keyword arguments to pass to the stream constructor.

    Returns:
        The stream if output_path is None, otherwise the output_path.
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

    resources, renderer, renderer_kwargs, kwargs = stream_cls._expand_from_any(
        resources, renderer, renderer_kwargs or {}, **kwargs
    )

    stream = stream_cls(
        resources=resources,
        iterables=iterables,
        renderer=renderer,
        renderer_kwargs=renderer_kwargs,
        **kwargs,
    )

    if output_path:
        return stream.write(output_path=output_path)
    return stream


def connect(
    streams: list[AnyStream | GifStream | Mp4Stream],
    output_path: str | Path | None = None,
) -> ConnectedStreams | Path:
    """
    Connect multiple streams into a single stream.

    Args:
        streams: The streams to connect.

    Returns:
        The connected streams if output_path is None, otherwise the output_path.
    """
    stream = ConnectedStreams(streams=streams)
    if output_path:
        return stream.write(output_path=output_path)
    return stream
