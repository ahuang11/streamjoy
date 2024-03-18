# 🌈 streamjoy 😊

[![build](https://github.com/ahuang11/streamjoy/workflows/Build/badge.svg)](https://github.com/ahuang11/streamjoy/actions) [![codecov](https://codecov.io/gh/ahuang11/streamjoy/branch/master/graph/badge.svg)](https://codecov.io/gh/ahuang11/streamjoy) [![PyPI version](https://badge.fury.io/py/streamjoy.svg)](https://badge.fury.io/py/streamjoy)

---

## 🔥 Enjoy animating!

Streamjoy merges images into animations--meticulously designed with sensible defaults and tested to ensure maximum enjoyment and minimal effort.

It cuts down the boilerplate and time to work on animations, and it's simple to start with just a few lines of code.

Install it with pip.
```python
pip install streamjoy
```

## 🛠️ Built-in features

🌐 Animate from URLs, files, and datasets
🎨 Render images with default or custom renderers
🎬 Provide context with a short intro splash
⏸ Add pauses at the beginning, end, or between frames
⚡ Execute read, render, write in parallel
🔗 Connect multiple animations together

## 🚀 Quick start

Stream from a list of images--local files work too!

```python
from streamjoy import stream

URL_FMT = "https://www.goes.noaa.gov/dimg/jma/fd/vis/{i}.gif"
resources = [URL_FMT.format(i=i) for i in range(1, 11)]
stream(resources, uri="goes.gif")  # .gif and .mp4 supported
```

Specify a few more keywords to:

1. add an intro title and subtitle
2. adjust the pauses

```python
from streamjoy import stream

URL_FMT = "https://www.goes.noaa.gov/dimg/jma/fd/vis/{i}.gif"
resources = [URL_FMT.format(i=i) for i in range(1, 11)]
himawari_stream = stream(
    resources,
    intro_title="Himawari Visible",
    intro_subtitle="10 Hours Loop",
    intro_pause=1,
    ending_pause=1
)
himawari_stream
```

Preview the inputs by excluding `uri`.

```yaml
<AnyStream>
---
Output:
  max_frames: 50
  fps: 10
  display: True
  scratch_dir: streamjoy_scratch
  in_memory: False
---
Intro:
  intro_title: Himawari Visible
  intro_subtitle: 10 Hours Loop
  intro_watermark: made with streamjoy
  intro_pause: 1
  intro_background: black
---
Client:
  batch_size: 10
  processes: True
  threads_per_worker: None
---
Resources: (10 frames to stream)
  https://www.goes.noaa.gov/dimg/jma/fd/vis/1.gif
  ...
  https://www.goes.noaa.gov/dimg/jma/fd/vis/10.gif
---
```

Then, simply add back `uri` or call the `write` method to save the animation!

```python
himawari_stream.write("goes_custom.gif")
```

Connect multiple streams together to provide further context.

```python
from streamjoy import stream, connect

URL_FMTS = {
    "visible": "https://www.goes.noaa.gov/dimg/jma/fd/vis/{i}.gif",
    "infrared": "https://www.goes.noaa.gov/dimg/jma/fd/rbtop/{i}.gif",
}
visible_stream = stream(
    [URL_FMTS["visible"].format(i=i) for i in range(1, 11)],
    intro_title="Himawari Visible",
    intro_subtitle="10 Hours Loop",
)
infrared_stream = stream(
    [URL_FMTS["infrared"].format(i=i) for i in range(1, 11)],
    intro_title="Himawari Infrared",
    intro_subtitle="10 Hours Loop",
)
connect([visible_stream, infrared_stream], uri="goes_connected.gif")
```

You can also render images directly from datasets, either through a custom renderer or a built-in one, and they'll also run in parallel!

```python
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from streamjoy import stream
from streamjoy.wrappers import wrap_matplotlib

@wrap_matplotlib()
def plot(da, central_longitude, **plot_kwargs):
    time = da["time"].dt.strftime("%b %d %Y").values.item()
    projection = ccrs.Orthographic(central_longitude=central_longitude)
    subplot_kw = dict(projection=projection, facecolor="gray")
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=subplot_kw)
    im = da.plot(ax=ax, transform=ccrs.PlateCarree(), add_colorbar=False, **plot_kwargs)
    ax.set_title(f"Sea Surface Temperature Anomaly\n{time}", loc="left", transform=ax.transAxes)
    ax.set_title("Source: NOAA OISST v2.1", loc="right", size=5, y=-0.01)
    ax.set_title("", loc="center")  # suppress default title
    plt.colorbar(im, ax=ax, label="°C", shrink=0.8)
    return fig

url = (
  "https://www.ncei.noaa.gov/data/sea-surface-temperature-"
  "optimum-interpolation/v2.1/access/avhrr/201008/"
)
stream(
    url,
    np.linspace(-140, -150, 30),  # iterables; central longitude per frame (30 frames)
    renderer=plot,  # base stream kwargs
    uri="oisst.mp4",
    pattern="oisst-avhrr-v02r01.*.nc",  # Mp4Stream.from_url kwargs
    max_files=30,
    cmap="RdBu_r", # plot kwargs to be forwarded
    vmin=-5,
    vmax=5,
)
```

Read the full docs [here](https://ahuang11.github.io/streamjoy/).

---

❤️ Made with considerable passion.
