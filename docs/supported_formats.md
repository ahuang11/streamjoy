
streamjoy supports a variety of input types!

### 📋 List

```python
from streamjoy import stream

URL_FMT = "https://noaadata.apps.nsidc.org/NOAA/G02135/north/daily/images/2024/01_Jan/N_202401{day:02d}_conc_v3.0.png"

stream([URL_FMT.format(day=day) for day in range(1, 31)], uri="2024_jan_sea_ice.gif")
```

### 📁 Directory

```python
from streamjoy import stream

URL_DIR = "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/Dailies/surface/"

stream(URL_DIR, uri="air_temperature.mp4", pattern="air.sig995.194*.nc")
```

### 🐼 Pandas DataFrame

```python
from streamjoy import stream
import pandas as pd

df = pd.read_csv(
    "https://raw.githubusercontent.com/franlopezguzman/gapminder-with-bokeh/master/gapminder_tidy.csv"
).set_index("Year")
df = df.query("Country in ['United States', 'China', 'South Africa']")
stream(df, uri="gapminder.mp4", groupby="Country", title="{Year}")
```

### 🗄️ XArray Dataset

```python
from streamjoy import stream
import xarray as xr

ds = xr.tutorial.open_dataset("air_temperature").isel(time=slice(0, 100))
stream(ds, uri="air_temperature.mp4", cmap="RdBu_r")
```

### 📊 HoloViews HoloMap or DynamicMap

```python
import xarray as xr
import hvplot.xarray
from streamjoy import stream

ds = xr.tutorial.open_dataset("rasm").isel(time=slice(10))
stream(ds.hvplot.image("x", "y"), uri="rasm.mp4")  
```
