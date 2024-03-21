# Sine wave

<img src="https://github.com/ahuang11/streamjoy/assets/15331990/61f103dc-5c6d-4957-a95c-5f66d6e0d71a" width="500">

Example of how to use `stream` alongside a custom `renderer` to create a sine wave animation.

Highlights:

- Uses `wrap_matplotlib` to automatically handle saving and closing the figure.
- Uses a custom `renderer` function to create each frame of the animation.

```python
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

if __name__ == "__main__":
    stream(list(range(10)), uri="sine_wave.gif", renderer=plot_frame)
```
