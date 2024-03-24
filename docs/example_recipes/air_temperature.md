# Air temperature

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
