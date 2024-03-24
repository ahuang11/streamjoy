# Air temperature

<video controls="true" allowfullscreen="true">
<source src="https://github.com/ahuang11/streamjoy/assets/15331990/cf7d8849-0c1c-4f8b-9592-a29040e2d30f" type="video/mp4">
</video>

Super barebones example of rendering air temperature data from xarray.

Highlights:

- Imports `streamjoy.xarray` to use the `stream` accessor.
- Passes the `uri` to `stream` as the first argument to save the animation to disk.

```python
import xarray as xr
import streamjoy.xarray

if __name__ == "__main__":
    ds = xr.tutorial.open_dataset("air_temperature")
    ds.streamjoy("air_temperature.mp4")
```
