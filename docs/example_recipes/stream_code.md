# Stream code

<img src="https://github.com/ahuang11/streamjoy/assets/15331990/4c7f4740-e407-4d86-84d6-b83582887f75" width="500">

Generates an animation of a code snippet being written character by character.

Highlights:

- Uses a custom `renderer` function to create each frame of the animation.
- Propagates `formatter`, `max_line_length`, and `max_line_number` to the custom `renderer` function.

```python
from textwrap import dedent

import numpy as np
from PIL import Image, ImageDraw
from pygments import lex
from pygments.formatters import ImageFormatter
from pygments.lexers import get_lexer_by_name
from streamjoy import stream

def _custom_format(
    formatter: ImageFormatter,
    tokensource: list[tuple],
    max_line_length: int = None,
    max_line_number: int = None,
) -> Image:
    formatter._create_drawables(tokensource)
    formatter._draw_line_numbers()
    max_line_length = max_line_length or formatter.maxlinelength
    max_line_number = max_line_number or formatter.maxlineno

    image = Image.new(
        "RGB",
        formatter._get_image_size(max_line_length, max_line_number),
        formatter.background_color,
    )
    formatter._paint_line_number_bg(image)
    draw = ImageDraw.Draw(image)
    # Highlight
    if formatter.hl_lines:
        x = (
            formatter.image_pad
            + formatter.line_number_width
            - formatter.line_number_pad
            + 1
        )
        recth = formatter._get_line_height()
        rectw = image.size[0] - x
        for linenumber in formatter.hl_lines:
            y = formatter._get_line_y(linenumber - 1)
            draw.rectangle([(x, y), (x + rectw, y + recth)], fill=formatter.hl_color)
    for pos, value, font, text_fg, text_bg in formatter.drawables:
        if text_bg:
            text_size = draw.textsize(text=value, font=font)
            draw.rectangle(
                [pos[0], pos[1], pos[0] + text_size[0], pos[1] + text_size[1]],
                fill=text_bg,
            )
        draw.text(pos, value, font=font, fill=text_fg)
    return np.asarray(image)

def render_frame(
    code: str,
    formatter: ImageFormatter,
    max_line_length: int = None,
    max_line_number: int = None,
) -> Image:
    lexer = get_lexer_by_name("python")
    return _custom_format(
        formatter,
        lex(code, lexer),
        max_line_length=max_line_length,
        max_line_number=max_line_number,
    )

if __name__ == "__main__":
    code = dedent(
        """
        import matplotlib.pyplot as plt
        import numpy as np
    
        from streamjoy import stream, wrap_matplotlib
    
        @wrap_matplotlib()
        def plot_frame(i):
            x = np.linspace(0, 2, 1000)
            y = np.sin(2 * np.pi * (x - 0.01 * i))
            fig, ax = plt.subplots()
            ax.plot(x, y)
            return fig
    
        stream(list(range(10)), uri="sine_wave.mp4", renderer=plot_frame)
        """
    )

    formatter = ImageFormatter(
        image_format="gif",
        line_pad=8,
        line_number_bg=None,
        line_number_fg=None,
        encoding="utf-8",
    )
    longest_line = max(code.splitlines(), key=len) + " " * 12
    max_line_length, _ = formatter.fonts.get_text_size(longest_line)
    max_line_number = code.count("\n") + 1
    items = [code[:i] for i in range(0, len(code) + 3, 3)]

    stream(
        items,
        ending_pause=20,
        uri="stream_code.gif",
        renderer=render_frame,
        formatter=formatter,
        max_line_length=max_line_length,
        max_line_number=max_line_number,
    )
```
