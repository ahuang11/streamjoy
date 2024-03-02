config = {
    "fps": 15,
    "max_urls": 1,
    "batch_size": 10,
    "max_frames": 100,
    "max_open_warning": 100,
    "logging_success_level": 25,
    "logging_level": 25,
    "logging_format": "[%(levelname)s] %(asctime)s: %(message)s",
    "logging_datefmt": "%I:%M%p",
    "logging_warning_color": "\x1b[31;1m",
    "logging_success_color": "\x1b[32;1m",
    "logging_reset_color": "\x1b[0m",
    "scratch_dir": "streamjoy_scratch",
    "output_path": "streamjoy",
    "mp4_codec": "libx264",
    "gif_loop": 0,
    "gif_pause": 4,
}

obj_readers = {
    "xarray.Dataset": "from_xarray",
    "xarray.DataArray": "from_xarray",
    "pandas.DataFrame": "from_pandas",
    "pandas.Series": "from_pandas",
    "holoviews.DynamicMap": "from_holoviews",
    "holoviews.HoloMap": "from_holoviews",
}

file_readers = {
    ".nc": {"import_path": "xarray.open_mfdataset", "batched": True},
    ".nc4": {"import_path": "xarray.open_mfdataset", "batched": True},
    ".zarr": {"import_path": "xarray.open_zarr", "batched": True},
    ".grib": {
        "import_path": "xarray.open_mfdataset",
        "kwargs": {"engine": "cfgrib"},
        "batched": True,
    },
    ".csv": {"import_path": "pandas.read_csv", "batched": False},
    ".json": {"import_path": "pandas.read_json", "batched": False},
    ".html": {"import_path": "pandas.read_html", "batched": False},
}
