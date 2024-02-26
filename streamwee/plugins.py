# ALL_PLUGINS = {
#     "xarray": {
#         "package": "xarray",
#         "acronym": "xr",
#         "types": ("xr.Dataset", "xr.DataArray"),
#     },
#     "pandas": {
#         "package": "pandas",
#         "acronym": "pd",
#         "types": ("pd.DataFrame", "pd.Series"),
#     },
#     "httpx": {
#         "package": "httpx",
#         "acronym": "httpx",
#         "types": ("httpx.Response",),
#     },
# }


# available_plugins = {}
# for plugin in ("xarray", "pandas", "httpx"):
#     try:
#         available_plugins[plugin] = {
#             "package": __import__(plugin),
#             "types":
#         }
#     except ImportError:
#         pass
