from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Literal

from .streams import AnyStream, ConnectedStreams, GifStream, Mp4Stream


def stream(
    resources: Any,
    uri: str | Path | BytesIO | None = None,
    renderer: Callable | None = None,
    renderer_iterables: list | None = None,
    renderer_kwargs: dict | None = None,
    extension: Literal[".mp4", ".gif"] | None = None,
    **kwargs: dict[str, Any],
) -> AnyStream | GifStream | Mp4Stream | Path:
    """
    Create a stream from the given resources.

    Args:
        resources: The resources to create a stream from.
        uri: The destination to write the stream to. If None, the stream is returned.
        renderer: The renderer to use. If None, the default renderer is used.
        renderer_iterables: Additional positional arguments to map over the renderer.
        renderer_kwargs: Additional keyword arguments to pass to the renderer.
        extension: The extension to use; useful if uri is a file-like object.
        **kwargs: Additional keyword arguments to pass.

    Returns:
        The stream if uri is None, otherwise the uri.
    """
    if isinstance(uri, str):
        uri = Path(uri)

    extension = extension or (uri and uri.suffix)
    if extension not in (None, ".mp4", ".gif"):
        raise ValueError(f"Unsupported extension: {extension}")

    if extension == ".mp4":
        stream_cls = Mp4Stream
    elif extension == ".gif":
        stream_cls = GifStream
    else:
        stream_cls = AnyStream

    resources, renderer, renderer_iterables, renderer_kwargs, kwargs = (
        stream_cls._expand_from_any(
            resources, renderer, renderer_iterables, renderer_kwargs or {}, **kwargs
        )
    )

    stream = stream_cls(
        resources=resources,
        renderer=renderer,
        renderer_iterables=renderer_iterables,
        renderer_kwargs=renderer_kwargs,
        **kwargs,
    )

    if uri:
        return stream.write(uri=uri, extension=extension)
    return stream


def connect(
    streams: list[AnyStream | GifStream | Mp4Stream],
    uri: str | Path | BytesIO | None = None,
) -> ConnectedStreams | Path:
    """
    Connect hetegeneous streams into a single stream.

    Unlike `stream.join`, this function can connect streams
    with unique params, such as different renderers.

    Args:
        streams: The streams to connect.
        uri: The destination to write the connected streams to.
            If None, the connected streams are returned.

    Returns:
        The connected streams if uri is None, otherwise the uri.
    """
    stream = ConnectedStreams(streams=streams)
    if uri:
        return stream.write(uri=uri)
    return stream
