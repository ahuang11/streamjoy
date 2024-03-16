from pathlib import Path
import pytest
import xarray as xr
import pandas as pd
from streamjoy._utils import get_distributed_client


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


@pytest.fixture(autouse=True, scope="session")
def client():
    return get_distributed_client()
