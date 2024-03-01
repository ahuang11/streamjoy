try:
    import pandas as pd
except Exception as exc:
    raise ImportError(
        "Could not patch streamjoy API onto pandas. Pandas could not be imported."
    ) from exc

from .core import stream


def patch(name="streamjoy"):
    _patch_plot = lambda self, *args, **kwargs: stream(self, *args, **kwargs)
    _patch_plot.__doc__ = stream.__doc__
    setattr(pd.DataFrame, name, _patch_plot)
    setattr(pd.Series, name, _patch_plot)


patch()
