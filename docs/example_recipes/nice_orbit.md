# Nice orbit

<video controls="true" allowfullscreen="true">
<source src="https://github.com/ahuang11/streamjoy/assets/15331990/775fa0ff-540c-4e48-b751-b68823ec511b" type="video/mp4">
</video>

Creates a visually appealing, nice orbits of a 2d dynamical system.

Code adapted from [Nice_orbits.ipynb](https://github.com/profConradi/Python_Simulations/blob/main/Nice_orbits.ipynb).
All credits go to [Simone Conradi](https://github.com/profConradi); the only addition here was wrapping the code into a function and using `streamjoy` to create the animation. Please consider giving the [Python_Simulations](https://github.com/profConradi/Python_Simulations/tree/main) repo a star!

Highlights:

- Uses `wrap_matplotlib` to automatically handle saving and closing the figure.
- Uses a custom `renderer` function to create each frame of the animation.

```python
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from streamjoy import stream, wrap_matplotlib

@njit
def meshgrid(x, y):
    """
    This function replace np.meshgrid that is not supported by numba
    """
    xx = np.empty(shape=(x.size, y.size), dtype=x.dtype)
    yy = np.empty(shape=(x.size, y.size), dtype=y.dtype)
    for j in range(y.size):
        for k in range(x.size):
            xx[j, k] = k  # change to x[k] if indexing xy
            yy[j, k] = j  # change to y[j] if indexing xy
    return xx, yy

@njit
def calc_orbit(n_points, a, b, n_iter):
    """
    This function calculate orbits in a vectorized fashion.

    -n_points: lattice of initial conditions, n_points x n_points in [-1,1]x[-1,1]
    -a: first parameter of the dynamical system
    -b: second parameter of the dynamical system
    -n_iter: number of iterations

    Return: two ndarrays: x and y coordinates of every point of every orbit.
    """
    area = [[-1, 1], [-1, 1]]
    x = np.linspace(area[0][0], area[0][1], n_points)
    y = np.linspace(area[1][0], area[1][1], n_points)
    xx, yy = meshgrid(x, y)
    l_cx, l_cy = np.zeros(n_iter * n_points**2), np.zeros(n_iter * n_points**2)
    for i in range(n_iter):
        xx_new = np.sin(xx**2 - yy**2 + a)
        yy_new = np.cos(2 * xx * yy + b)
        xx = xx_new
        yy = yy_new
        l_cx[i * n_points**2 : (i + 1) * n_points**2] = xx.flatten()
        l_cy[i * n_points**2 : (i + 1) * n_points**2] = yy.flatten()
    return l_cx, l_cy

@wrap_matplotlib()
def plot_frame(n):
    l_cx, l_cy = calc_orbit(n_points, a + 0.002 * n, b - 0.001 * n, n)
    area = [[-1, 1], [-1, 1]]
    h, _, _ = np.histogram2d(l_cx, l_cy, bins=3000, range=area)
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax.imshow(np.log(h + 1), vmin=0, vmax=5, cmap="magma")
    plt.xticks([]), plt.yticks([])
    return fig

if __name__ == "__main__":
    n_points = 500
    a, b = 5.48, 4.28
    stream(np.arange(1, 100).tolist(), renderer=plot_frame, uri="nice_orbit.mp4")
```
