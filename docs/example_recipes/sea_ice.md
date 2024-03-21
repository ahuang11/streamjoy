# Sea ice

<video controls="true" allowfullscreen="true">
<source src="https://github.com/ahuang11/streamjoy/assets/15331990/f141fe3e-1435-4ddb-a8cc-f09cf1850c6e" type="video/mp4">
</video>

Compares sea ice concentration data from the NOAA G02135 dataset for August 15th in 1989 and 2023.

Highlights:

- Downloads images directly from the NSIDC FTP server.
- Uses `connect` to concatenate the two homogeneous streams together (same keyword arguments, different resources).
- Uses `pattern` to filter for only the sea ice concentration images.
- Uses `intro_title` and `intro_subtitle` to provide context at the beginning of the animation.

```python
from streamjoy import stream, connect

connect(
    [
        stream(
            f"https://noaadata.apps.nsidc.org/NOAA/G02135/north/daily/images/{year}/08_Aug/",
            pattern=f"N_*_conc_v3.0.png",
            intro_title=f"August 15, {year}",
            intro_subtitle="Sea Ice Concentration",
            max_files=31,
        )
        for year in [1989, 2023]
    ]
).write("sea_ice.mp4")
```