from pathlib import Path

import hvplot.xarray  # noqa: F401
import pandas as pd
import pytest
import xarray as xr

from streamjoy._utils import get_distributed_client
from streamjoy.settings import config

DATA_DIR = Path(__file__).parent / "data"
NC_PATH = DATA_DIR / "air.nc"
ZARR_PATH = DATA_DIR / "air.zarr"
CSV_PATH = DATA_DIR / "gapminder.csv"
PARQUET_PATH = DATA_DIR / "gapminder.parquet"


@pytest.fixture
def ds():
    return xr.open_zarr(ZARR_PATH)


@pytest.fixture
def df():
    return pd.read_parquet(PARQUET_PATH)


@pytest.fixture
def dmap(ds):
    return ds.hvplot("lon", "lat", dynamic=True)


@pytest.fixture
def hmap(ds):
    return ds.hvplot("lon", "lat", dynamic=False)


@pytest.fixture(autouse=True, scope="session")
def client():
    return get_distributed_client()


@pytest.fixture(autouse=True, scope="session")
def default_config():
    config["max_frames"] = 3
    config["max_files"] = 2
    config["ending_pause"] = 0


@pytest.fixture(scope="session")
def data_dir():
    return DATA_DIR
