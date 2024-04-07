# NMME forecast

<iframe
    width="100%" height="800px"
    src="../assets/nmme_forecast.html"
    frameborder="0"
    marginheight="0"
    marginwidth="0">
</iframe>

Highlights:

- Writes to memory to later use in another Panel component
- Sets `extension` to hint at the desired output format
- Appends two streams in a `Tabs` layout
- Links the Players' value from the first tab to the second tab

```python hl_lines="30 36 37 38"
import panel as pn
import pandas as pd
from streamjoy import stream

pn.extension()

URL_FMT = (
    "https://www.cpc.ncep.noaa.gov/products/NMME/archive/{dt:%Y%m}0800/"
    "current/images/NMME_ensemble_{var}_us_lead{i}.png"
)

VARS = {
    "prate": "Precipitation Rate",
    "tmp2m": "2m Temperature",
}
LEADS = 7

var_tabs = pn.Tabs()
for var in VARS.keys():
    dt_range = [
        pd.to_datetime("2024-03-08") - pd.DateOffset(months=lead)
        for lead in range(LEADS)
    ]
    urls = [
        URL_FMT.format(i=i, dt=dt, var=var)
        for i, dt in enumerate(dt_range, 1)
    ]
    col = stream(
        urls,
        extension=".html",
        fps=1,
        ending_pause=0,
        display=False,
        sizing_mode="stretch_width",
        height=400,
    ).write()
    var_tabs.append((VARS[var], col))
var_tabs[0][1].jslink(var_tabs[1][1], value="value")
var_tabs.save("nmme_forecast.html")
```