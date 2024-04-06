# Supported formats

StreamJoy supports a variety of input types!

## 📋 List of Images, GIFs, Videos, or URLs

```python
from streamjoy import stream

URL_FMT = "https://noaadata.apps.nsidc.org/NOAA/G02135/north/daily/images/2024/01_Jan/N_202401{day:02d}_conc_v3.0.png"

stream([URL_FMT.format(day=day) for day in range(1, 31)], uri="2024_jan_sea_ice.mp4")
```
<video controls="true" allowfullscreen="true">
<source src="https://github.com/ahuang11/streamjoy/assets/15331990/7c933cd4-aa15-461a-af79-f508d9d76aa5" type="video/mp4">
</video>

## 📁 Directory of Images, GIFs, Videos, or URLs

```python
from streamjoy import stream

URL_DIR = "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/Dailies/surface/"

stream(URL_DIR, uri="air_temperature.mp4", pattern="air.sig995.194*.nc")
```

<video controls="true" allowfullscreen="true">
<source src="https://github.com/ahuang11/streamjoy/assets/15331990/93cb0c1b-46d3-48e6-be2c-e3b1487f9117" type="video/mp4">
</video>

## 🧮 Numpy NdArray

```python
from streamjoy import stream
import imageio.v3 as iio

array = iio.imread("imageio:newtonscradle.gif")  # is a 4D numpy array
stream(array, max_frames=-1).write("newtonscradle.mp4")
```

<video controls="true" allowfullscreen="true">
<source src="https://github.com/ahuang11/streamjoy/assets/15331990/7687e951-654c-4719-b50a-4aabc0ddf2e4" type="video/mp4">
</video>

## 🐼 Pandas DataFrame or Series

!!! note "Additional Requirements"

    You will need to additionally install `pandas` and `matplotlib` to support this format:

    ```bash
    pip install pandas matplotlib
    ```

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

!!! note "Additional Requirements"

    You will need to additionally install `polars`, `pyarrow`, `hvplot`, `selenium`, and `webdriver-manager` to support this format:

    ```bash
    pip install polars pyarrow hvplot selenium webdriver-manager
    ```

    You must also have `firefox` or `chromedriver` installed on your system.

    ```bash
    conda install -c conda-forge firefox
    ```

```python
from streamjoy import stream
import polars as pl

df = pl.read_csv(
    "https://raw.githubusercontent.com/franlopezguzman/gapminder-with-bokeh/master/gapminder_tidy.csv"
)
df = df.filter(pl.col("Country").is_in(['United States', 'China', 'South Africa']))
stream(df, uri="gapminder.mp4", groupby="Country", title="{Year}")
```

<video controls="true" allowfullscreen="true">
<source src="https://github.com/ahuang11/streamjoy/assets/15331990/75531331-3974-46b8-8399-6dd14ad31f5c" type="video/mp4">
</video>

## 🗄️ XArray Dataset or DataArray

!!! note "Additional Requirements"

    You will need to additionally install `xarray` and `matplotlib` to support this format:

    ```bash
    pip install xarray matplotlib
    ```

    For this example, you will also need to install `pooch` and `netcdf4`:

    ```bash
    pip install pooch netcdf4
    ```

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

!!! note "Additional Requirements"

    You will need to additionally install `holoviews` to support this format:

    ```bash
    pip install holoviews
    ```

    For the bokeh backend, you will need to install `bokeh`, `selenium`, and `webdriver-manager`:

    ```bash
    pip install bokeh selenium webdriver-manager
    ```

    For the matplotlib backend, you will need to install `matplotlib`:

    ```bash
    pip install matplotlib
    ```

    For this example, you will also need to install `pooch`, `netcdf4, `hvplot`, and `xarray`:

    ```bash
    pip install pooch netcdf4 hvplot xarray
    ```

    You must also have `firefox` or `chromedriver` installed on your system.

    ```bash
    conda install -c conda-forge firefox
    ```

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
