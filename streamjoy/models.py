from __future__ import annotations

from typing import Any

import param
from PIL import ImageDraw, ImageFont

from . import _utils


class Paused(param.Parameterized):
    """
    A data model for pausing a stream.

    Expand Source code to see all the parameters and descriptions.
    """

    output = param.Parameter(doc="The output to pause for.")

    seconds = param.Integer(doc="The number of seconds to pause for.")

    def __init__(self, output: Any, seconds: int, **params):
        self.output = output
        self.seconds = seconds
        super().__init__(**params)


class ImageText(param.Parameterized):
    """
    A data model for rendering text on an image.

    Expand Source code to see all the parameters and descriptions.
    """

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
            params, self.param, config_prefix="image_text"
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
