# OISST globe


<video controls="true" allowfullscreen="true">
<source src="https://github.com/ahuang11/streamjoy/assets/15331990/a76d2f16-bea9-4168-9ba6-63dbc7967ae2" type="video/mp4">
</video>

Render sea surface temperature anomaly data from the NOAA OISST v2.1 dataset on a globe.

Highlights:

- Concatenates multiple homogeneous streams together (same keyword arguments, different resources) by summing them.
- Uses the built-in `default_xarray_renderer` under the hood
- Uses `renderer_kwargs` to pass keyword arguments to the underlying `ds.plot` method.

```python
import cartopy.crs as ccrs
from streamjoy import stream

YEAR = 2023
URL_FMT = "https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/{year}{month:02}/"

if __name__ == "__main__":
    streams = []
    for month in range(1, 13):
        url = URL_FMT.format(year=YEAR, month=month)
        streams.append(
            stream(
                url,
                pattern="oisst-avhrr-v02r01*.nc",
                var="anom",
                dim="time",
                max_files=29,
                max_frames=-1,
                renderer_kwargs=dict(
                    cmap="RdBu_r",
                    vmin=-5,
                    vmax=5,
                    subplot_kws=dict(
                        projection=ccrs.Orthographic(central_longitude=-150),
                        facecolor="gray",
                    ),
                    transform=ccrs.PlateCarree(),
                ),
            )
        )

    joined_stream = sum(streams)
    joined_stream.write("oisst_globe.mp4")
```