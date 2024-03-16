from __future__ import annotations

import param
from PIL import ImageDraw, ImageFont

from . import _utils


class Paused(param.Parameterized):

    output = param.Parameter()

    seconds = param.Integer()

    def __init__(self, output: str, seconds: int, **params):
        self.output = output
        self.seconds = seconds
        super().__init__(**params)


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
            params, self.param.values(), config_prefix="image_text"
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
