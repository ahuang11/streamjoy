# Supported formats

StreamJoy supports a variety of input types!

## 📋 List of Images or URLs

```python
from streamjoy import stream

URL_FMT = "https://noaadata.apps.nsidc.org/NOAA/G02135/north/daily/images/2024/01_Jan/N_202401{day:02d}_conc_v3.0.png"

stream([URL_FMT.format(day=day) for day in range(1, 31)], uri="2024_jan_sea_ice.mp4")
```
<video controls="true" allowfullscreen="true">
<source src="https://github.com/ahuang11/streamjoy/assets/15331990/7c933cd4-aa15-461a-af79-f508d9d76aa5" type="video/mp4">
</video>

## 📁 Directory of Images or URLs

```python
from streamjoy import stream

URL_DIR = "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/Dailies/surface/"

stream(URL_DIR, uri="air_temperature.mp4", pattern="air.sig995.194*.nc")
```

<video controls="true" allowfullscreen="true">
<source src="https://github.com/ahuang11/streamjoy/assets/15331990/93cb0c1b-46d3-48e6-be2c-e3b1487f9117" type="video/mp4">
</video>

## 🐼 Pandas DataFrame or Series

```python
from streamjoy import stream
import pandas as pd

df = pd.read_csv(
    "https://raw.githubusercontent.com/franlopezguzman/gapminder-with-bokeh/master/gapminder_tidy.csv"
).set_index("Year")
df = df.query("Country in ['United States', 'China', 'South Africa']")
stream(df, uri="gapminder.mp4", groupby="Country", title="{Year}")
```

<video controls="true" allowfullscreen="true">
<source src="https://github.com/ahuang11/streamjoy/assets/15331990/be0fc06c-c821-4c45-91a3-8c898e730851" type="video/mp4">
</video>

## 🐻‍❄️ Polars DataFrame
```python
from streamjoy import stream
import polars as pl

df = pl.read_csv(
    "https://raw.githubusercontent.com/franlopezguzman/gapminder-with-bokeh/master/gapminder_tidy.csv"
)
df = df.query("Country in ['United States', 'China', 'South Africa']")
stream(df, uri="gapminder.mp4", groupby="Country", title="{Year}")
```


## 🗄️ XArray Dataset or DataArray

```python
from streamjoy import stream
import xarray as xr

ds = xr.tutorial.open_dataset("air_temperature").isel(time=slice(0, 100))
stream(ds, uri="air.mp4", cmap="RdBu_r")
```

<video controls="true" allowfullscreen="true">
<source src="https://github.com/ahuang11/streamjoy/assets/15331990/969b78e2-9996-4ed9-9596-9344fb0fab1f" type="video/mp4">
</video>

## 📊 HoloViews HoloMap or DynamicMap

```python
import xarray as xr
import hvplot.xarray
from streamjoy import stream

ds = xr.tutorial.open_dataset("rasm").isel(time=slice(10))
stream(ds.hvplot.image("x", "y"), uri="rasm.mp4")  
```

<video controls="true" allowfullscreen="true">
<source src="https://github.com/ahuang11/streamjoy/assets/15331990/696a33c9-4167-4f25-a912-4278353eea14" type="video/mp4">
</video>
