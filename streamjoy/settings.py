config = {
    # animation
    "fps": 5,
    "max_frames": 50,
    # dask
    "batch_size": 10,
    "processes": True,
    "threads_per_worker": None,
    # intro
    "intro_pause": 3,
    "intro_watermark": "made with streamjoy",
    "intro_background": "black",
    # from_url
    "max_urls": 1,
    # matplotlib
    "max_open_warning": 100,
    # output
    "in_memory": False,
    "scratch_dir": "streamjoy_scratch",
    "output_path": "streamjoy.mp4",
    # imageio
    "codec": "libx264",
    "loop": 0,
    "ending_pause": 3,
    # image text
    "image_text_font": "Avenir.ttc",
    "image_text_size": 20,
    "image_text_color": "white",
    "image_text_background": "black",
    # notebook
    "display": True,
    # logging
    "logging_success_level": 25,
    "logging_level": 25,
    "logging_format": "[%(levelname)s] %(asctime)s: %(message)s",
    "logging_datefmt": "%I:%M%p",
    "logging_warning_color": "\x1b[31;1m",
    "logging_success_color": "\x1b[32;1m",
    "logging_reset_color": "\x1b[0m",
}

obj_handlers = {
    "xarray.Dataset": "_expand_from_xarray",
    "xarray.DataArray": "_expand_from_xarray",
    "pandas.DataFrame": "_expand_from_pandas",
    "pandas.Series": "_expand_from_pandas",
    "holoviews": "_expand_from_holoviews",
}

file_handlers = {
    ".nc": {
        "import_path": "xarray.open_mfdataset",
    },
    ".nc4": {
        "import_path": "xarray.open_mfdataset",
    },
    ".zarr": {
        "import_path": "xarray.open_zarr",
    },
    ".grib": {
        "import_path": "xarray.open_mfdataset",
        "kwargs": {"engine": "cfgrib"},
    },
    ".csv": {
        "import_path": "pandas.read_csv",
    },
    ".json": {
        "import_path": "pandas.read_json",
    },
    ".html": {
        "import_path": "pandas.read_html",
    },
}
