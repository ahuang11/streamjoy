from PIL import Image, ImageDraw

from streamjoy.models import ImageText, Paused


class TestPaused:
    def test_paused_initialization(self):
        output = "some_output"
        seconds = 5
        paused_instance = Paused(output=output, seconds=seconds)
        assert paused_instance.output == output
        assert paused_instance.seconds == seconds


class TestImageText:
    def test_image_text_initialization(self):
        text = "Hello, World!"
        font = "Arial"
        size = 24
        color = "black"
        anchor = "mm"
        x = 100
        y = 100
        kwargs = {"width": 500}

        image_text_instance = ImageText(
            text=text,
            font=font,
            size=size,
            color=color,
            anchor=anchor,
            x=x,
            y=y,
            kwargs=kwargs,
        )

        assert image_text_instance.text == text
        assert image_text_instance.font == font
        assert image_text_instance.size == size
        assert image_text_instance.color == color
        assert image_text_instance.anchor == anchor
        assert image_text_instance.x == x
        assert image_text_instance.y == y
        assert image_text_instance.kwargs == kwargs

    def test_image_text_render(self):
        text = "Test Render"
        image_text_instance = ImageText(
            text=text, font="Arial", size=24, color="black", anchor="mm", x=50, y=50
        )
        img = Image.new("RGB", (100, 100), color="white")
        draw = ImageDraw.Draw(img)
        image_text_instance.render(draw)

        # Since it's difficult to assert the actual drawing, we check if the method runs without errors
        assert True  # This is a placeholder to indicate the test passed by reaching this point without errors
