# Best practices

## ⏸️ Take advantage of `Paused`, `intro_pause`, `ending_pause`

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

## 🎥 When to use `.mp4` vs `.gif`

If you need a comprehensive color palette, use `.mp4` as it supports more colors.

For automatic playing and looping, use `.gif`. To reduce the file size of the `.gif`, set `optimize=True`, which uses `pygifsicle` to reduce the file size.

## 📦 Wrap `streamjoy` functionality under `__name__ == "__main__"`

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

It's fine without it in notebooks though.

## ⚙️ Change the default `config` settings

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

## 🔧 Explicitly set keyword arguments

Much of StreamJoy is based on sensible defaults to get you started quickly, but you should override them.

For example, `max_frames` is set to 50 by default so you can quickly preview the animation. If you want to render the entire animation, set `max_frames=-1`.

StreamJoy will warn you on some settings if you don't override them:

```python
No 'max_frames' specified; using the default 50 / 100 frames. Pass `-1` to use all frames. Suppress this by passing 'max_frames'.
```

## 🧩 Use `processes=False` for rendering HoloViews objects

This is done automatically! However, in case there's an edge case, note that the kdims/vdims don't seem to carry over properly to the subprocesses when rendering HoloViews objects. It might complain that it can't find the desired dimensions.
