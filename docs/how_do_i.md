# How do I...

## 🖼️ Use all resources with `max_frames=-1`

By default, StreamJoy only renders the first 50 frames to prevent accidentally rendering a large dataset.

To render all frames, set `max_frames=-1`.

```python
from streamjoy import stream

stream(..., max_frames=-1)
```

## ⏸️ How to pause animations with `Paused`, `intro_pause`, `ending_pause`

Animations can be good, but sometimes you want to pause at various points of the animation to provide context or to emphasize a point.

To pause at a given frame using a custom `renderer`, wrap `Paused` around the output:

```python
from streamjoy import stream

def plot_frame(time)
    important_time = ...
    if time == some_time:
        return Paused(fig, seconds=3)
    else:
        return fig

stream(..., renderer=plot_frame)
```

Don't forget there's also `intro_pause` and `ending_pause` to pause at the beginning and end of the animation!

## 📊 Reduce boilerplate code with `wrap_*` decorators

If you're using a custom `renderer`, you can use `wrap_matplotlib` and `wrap_holoviews` to automatically handle saving and closing the figure.

```python
from streamjoy import stream, wrap_matplotlib

@wrap_matplotlib()
def plot_frame(time):
    ...

stream(..., renderer=plot_frame)
```

## 🗣️ Provide context with `intro_title` and `intro_subtitle`

Use `intro_title` and `intro_subtitle` to provide context at the beginning of the animation.

```python
from streamjoy import stream

stream(..., intro_title="Himawari Visible", intro_subtitle="10 Hours Loop")
```

## 💾 Write animation to memory instead of file

If you're just testing out the animation, you can save it to memory instead of to disk by calling write without specifying a uri.

```python
from streamjoy import stream

stream(...).write()
```

## 🚪 Use as a method of `pandas` and `xarray` objects

StreamJoy can be used directly from `pandas` and `xarray` objects as an accessor.

```python
import pandas as pd
import streamjoy.pandas

df = pd.DataFrame(...)

# equivalent to streamjoy.stream(df)
df.streamjoy(...)

# series also works!
df["col"].streamjoy(...)
```

```python
import xarray as xr
import streamjoy.xarray

ds = xr.Dataset(...)

# equivalent to streamjoy.stream(ds)
ds.streamjoy(...)

# dataarray also works!
ds["var"].streamjoy(...)
```

## ⛓️ Join streams with `sum` and `connect`

Use `sum` to join homogeneous streams, i.e. streams that have the same keyword arguments.

```python
from streamjoy import stream, sum

sum([stream(..., **same_kwargs) for i in range(10)])
```

Use `connect` to join heterogeneous streams, i.e. streams that have different keyword arguments, like different `intro_title` and `intro_subtitle`.

```python
from streamjoy import stream, connect

connect([stream(..., **kwargs1), stream(..., **kwargs2)])
```

## 🎥 Decide between writing as `.mp4` vs `.gif`

If you need a comprehensive color palette, use `.mp4` as it supports more colors.

For automatic playing and looping, use `.gif`. To reduce the file size of the `.gif`, set `optimize=True`, which uses `pygifsicle` to reduce the file size.

## 📦 Prevent `RuntimeError` by using `__name__ == "__main__"`

If you run a `.py` script without it, you might encounter the following `RuntimeError`:

```python
RuntimeError:
An attempt has been made to start a new process before the
current process has finished its bootstrapping phase.

This probably means that you are not using fork to start your
child processes and you have forgotten to use the proper idiom
in the main module:

    if __name__ == '__main__':
        freeze_support()
        ...

The "freeze_support()" line can be omitted if the program
is not going to be frozen to produce an executable.
```

To patch, simply wrap your `stream` call in `if __name__ == "__main__":`.

```python
if __name__ == "__main__":
    stream(...)
```

It's fine without it in notebooks though.

## ⚙️ Set your own default settings with `config`

StreamJoy uses a simple `config` dict to store settings. You can change the default settings by modifying the `streamjoy.config` object.

To see the available options:
```python
import streamjoy

print(streamjoy.config)
```

To change the settings, it's simply updating the key-value pair.
```python
import streamjoy

streamjoy.config["max_frames"] = -1
```

Be wary of completely overwriting the `config` object, as it might break the functionality; do not do this!
```python
import streamjoy

streamjoy.config = {"max_frames": -1}
```

## 🔧 Use custom values instead of the defaults

Much of StreamJoy is based on sensible defaults to get you started quickly, but you should override them.

For example, `max_frames` is set to 50 by default so you can quickly preview the animation. If you want to render the entire animation, set `max_frames=-1`.

StreamJoy will warn you on some settings if you don't override them:

```python
No 'max_frames' specified; using the default 50 / 100 frames. Pass `-1` to use all frames. Suppress this by passing 'max_frames'.
```

## 🧩 Render HoloViews objects with `processes=False`

This is done automatically! However, in case there's an edge case, note that the kdims/vdims don't seem to carry over properly to the subprocesses when rendering HoloViews objects. It might complain that it can't find the desired dimensions.

## 📚 Prevent flickering by setting `threads_per_worker`

Matplotlib is not always thread-safe, so if you're seeing flickering, set `threads_per_worker=1`.

```python
from streamjoy import stream

stream(..., threads_per_worker=1)
```

## 🖥️ Provide `client` if using a remote cluster

If you're using a remote cluster, specify the `client` argument to use the Dask client.

```python
from dask.distributed import Client

client = Client()
stream(..., client=client)
```

## 🪣 Read & write files on a remote filesystem with `fsspec_fs`

To read and write files on a remote filesystem, use `fsspec_fs` to specify the filesystem.

A scratch directory must be provided; be sure to prefix the bucket name.

```python
fs = fsspec.filesystem('s3', anon=False)
stream(..., fsspec_fs=fs, scratch_dir="bucket-name/streamjoy_scratch")
```

## 🚗 Use a custom webdriver to render HoloViews

By default, StreamJoy uses Firefox as the default headless webdriver to render HoloViews objects into images.

If you want to use Chrome instead, you can pass `webdriver="chrome"`.

If you want to use a different webdriver, you can pass a custom function to `webdriver`.

```python
def get_webdriver():
    from selenium.webdriver.firefox.options import Options
    from selenium.webdriver.firefox.webdriver import Service, WebDriver
    from webdriver_manager.firefox import GeckoDriverManager

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-extensions")
    executable_path = GeckoDriverManager().install()
    driver = WebDriver(
        service=Service(executable_path), options=options
    )
    return driver

stream(..., webdriver=get_webdriver)
```
