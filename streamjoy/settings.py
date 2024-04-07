config = {
    # animation
    "fps": 8,
    "max_frames": 50,
    # dask
    "batch_size": 10,
    "processes": True,
    "threads_per_worker": None,
    # intro
    "intro_pause": 2,
    "intro_watermark": "made with streamjoy",
    "intro_background": "black",
    # from_url
    "max_files": 2,
    # download
    "parent_depth": 4,
    # matplotlib
    "max_open_warning": 100,
    # holoviews
    "webdriver": "firefox",
    "num_retries": 5,
    # output
    "in_memory": False,
    "scratch_dir": "streamjoy_scratch",
    "uri": None,
    # imageio
    "codec": "libx264",
    "loop": 0,
    "ending_pause": 2,
    # gif
    "optimize": False,
    # image text
    "image_text_font": "Avenir.ttc",
    "image_text_size": 20,
    "image_text_color": "white",
    "image_text_background": "black",
    # notebook
    "display": True,
    # logging
    "logging_success_level": 25,
    "logging_level": 20,
    "logging_format": "[%(levelname)s] %(asctime)s: %(message)s",
    "logging_datefmt": "%I:%M%p",
    "logging_warning_color": "\x1b[31;1m",
    "logging_success_color": "\x1b[32;1m",
    "logging_reset_color": "\x1b[0m",
}

extension_handlers = {
    None: "AnyStream",
    ".mp4": "Mp4Stream",
    ".gif": "GifStream",
    ".html": "HtmlStream",
}

obj_handlers = {
    "xarray.Dataset": "serialize_xarray",
    "xarray.DataArray": "serialize_xarray",
    "pandas.DataFrame": "serialize_pandas",
    "pandas.Series": "serialize_pandas",
    "holoviews": "serialize_holoviews",
    "polars.DataFrame": "serialize_polars",
    "numpy.ndarray": "serialize_numpy",
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
        "concat_path": "pandas.concat",
    },
    ".parquet": {
        "import_path": "pandas.read_parquet",
        "concat_path": "pandas.concat",
    },
    ".html": {
        "import_path": "pandas.read_html",
        "concat_path": "pandas.concat",
    },
}
