import re
from io import BytesIO

try:
    import param
    import panel as pn

    pn.extension(notifications=True)
except ImportError:
    raise ImportError(
        "StreamJoy UI additionally requires panel"
        "run `pip install 'streamjoy[ui]'` to install."
    )

from .core import stream


class App(pn.viewable.Viewer):

    url = param.String(
        label="URL",
        default="https://noaadata.apps.nsidc.org/NOAA/G02135/north/daily/images/2024/01_Jan/",
    )

    max_files = param.Integer(bounds=(0, 1000), default=10)

    pattern = param.String(default="N_202401{DAY:02d}_conc_v3.0.png")

    pattern_inputs_start = param.Integer(bounds=(0, 1000), default=1)

    pattern_inputs_end = param.Integer(bounds=(0, 1000), default=10)

    pattern_inputs = param.Dict()

    extension = param.Selector(objects=[".gif", ".html"], default=".html")

    def __init__(self, **params):
        super().__init__(**params)
        url_input = pn.widgets.TextInput.from_param(
            self.param.url, placeholder="Enter URL"
        )
        max_files_input = pn.widgets.Spinner.from_param(
            self.param.max_files, name="Max Files"
        )
        pattern_input = pn.widgets.TextInput.from_param(
            self.param.pattern, placeholder="Enter pattern (e.g. *.png or {0}.png)"
        )
        pattern_inputs_simple = pn.WidgetBox(
            pn.widgets.Spinner.from_param(
                self.param.pattern_inputs_start, name="Start of {}"
            ),
            pn.widgets.Spinner.from_param(
                self.param.pattern_inputs_end, name="End of {}"
            ),
        )
        pattern_inputs_editor = pn.widgets.JSONEditor.from_param(
            self.param.pattern_inputs,
            mode="form",
            value={"i": [0]},
            sizing_mode="stretch_width",
            search=False,
            menu=False,
        )
        self._pattern_inputs_tabs = pn.Tabs(
            ("Simple", pattern_inputs_simple),
            # ("Editor", pattern_inputs_editor),
        )
        self._pattern_preview = pn.pane.HTML("<b>Preview</b>: ")
        self._pattern_view = pn.Column(
            pn.pane.HTML("<b>Pattern Inputs</b>"),
            self._pattern_inputs_tabs,
            self._pattern_preview,
        )
        input_widgets = pn.Card(
            url_input,
            pattern_input,
            max_files_input,
            self._pattern_view,
            title="Inputs",
            sizing_mode="stretch_width"
        )
        submit_button = pn.widgets.Button(
            on_click=self._on_submit,
            name="Submit",
            sizing_mode="stretch_width",
            button_type="success",
        )
        self._download_button = pn.widgets.FileDownload(
            filename="streamjoy.html",
            callback=self._download,
            sizing_mode="stretch_width",
            button_type="primary",
            disabled=True,
        )
        extension_input = pn.widgets.Select.from_param(
            self.param.extension, sizing_mode="stretch_width"
        )
        self._sidebar = pn.Column(
            pn.Row(submit_button, self._download_button),
            extension_input,
            input_widgets,
        )
        self._main = pn.Column()
        self._dashboard = pn.template.FastListTemplate(
            title="StreamJoy",
            sidebar=[self._sidebar],
            main=[self._main],
        )
        self._update_pattern_preview()

    def _extract_templates(self, pattern):
        pattern_formats = re.search(r"{(\w+)", pattern)
        return pattern_formats

    @param.depends("pattern", watch=True)
    def _update_pattern_inputs(self):
        pattern = self.pattern
        pattern_formats = self._extract_templates(pattern)

        if pattern_formats is not None:
            self._pattern_view.visible = True
            pattern_inputs_simple = self._pattern_inputs_tabs[0]
            # pattern_inputs_editor = self._pattern_inputs_tabs[1]
            try:
                pattern_formats.group(2)
                pn.state.notifications.error(f"Only one pattern format is allowed.")
                pattern_inputs_simple.disabled = True
                # pattern_inputs_editor.disabled = True
            except IndexError:
                pass
            pattern_format_key = pattern_formats.group(1)
            pattern_inputs_simple.disabled = False
            # pattern_inputs_editor.disabled = False
            pattern_inputs_simple[0].name = f"Start of {pattern_format_key}"
            # pattern_inputs_simple[1].name = f"End of {pattern_format_key}"
        else:
            self._pattern_view.visible = False

    @param.depends("pattern", "pattern_inputs_start", "pattern_inputs_end", watch=True)
    def _update_pattern_preview(self):
        pattern = self.pattern
        pattern_formats = self._extract_templates(pattern)
        if pattern_formats is not None:
            pattern_format_key = pattern_formats.group(1)
            pattern_inputs_start = self.pattern_inputs_start
            pattern_inputs_end = self.pattern_inputs_end
            pattern_start = pattern.format(**{pattern_format_key: pattern_inputs_start})
            pattern_end = pattern.format(**{pattern_format_key: pattern_inputs_end})
            self._pattern_preview.object = (
                f"<b>Preview</b>:<br>"
                f"{pattern_start}"
                f"<br>...to...<br>"
                f"{pattern_end}"
            )

    def _on_submit(self, event):
        with self._sidebar.param.update(loading=True):
            if self.url:
                stream_kwargs = {}
                if self._pattern_view.visible:
                    pattern = self.pattern
                    pattern_formats = self._extract_templates(pattern)
                    if pattern_formats is not None:
                        url = self.url
                        if not url.endswith("/"):
                            url += "/"
                        pattern_format_key = pattern_formats.group(1)
                        pattern_inputs_start = self.pattern_inputs_start
                        pattern_inputs_end = self.pattern_inputs_end
                        resources = []
                        for pattern_input in range(
                            pattern_inputs_start, pattern_inputs_end + 1
                        ):
                            resource = self.url + pattern.format(
                                **{pattern_format_key: pattern_input}
                            )
                            resources.append(resource)
                else:
                    resources = self.url
                    stream_kwargs["pattern"] = self.pattern
                    stream_kwargs["max_files"] = self.max_files

                if self.extension == ".html":
                    stream_kwargs["ending_pause"] = 0
                
                output = stream(
                    resources, extension=self.extension, **stream_kwargs
                ).write()

                if self.extension == ".html":
                    self._main.objects = [output]
                    buf = BytesIO()
                    output.save(buf)
                    self._buf = buf
                else:
                    self._main.objects = [pn.pane.GIF(output)]
                    self._buf = output
                self._download_button.disabled = False

    def _download(self):
        self._download_button.filename = f"streamjoy{self.extension}"
        self._buf.seek(0)
        return self._buf

    def serve(self, port: int = 8888, show: bool = True, **kwargs):
        pn.serve(self.__panel__(), port=port, show=show, **kwargs)

    def __panel__(self):
        return self._dashboard
