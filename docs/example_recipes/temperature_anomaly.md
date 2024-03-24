# Temperature anomaly

<video controls="true" allowfullscreen="true">
<source src="https://github.com/ahuang11/streamjoy/assets/15331990/069b1826-de92-4643-8be5-6d5a5301d11e" type="video/mp4">
</video>

Shows the global temperature anomaly from 1995 to 2024 using the HadCRUT5 dataset. The video pauses at notable dates.

Highlights:

- Uses `wrap_matplotlib` to automatically handle saving and closing the figure.
- Uses a custom `renderer` function to create each frame of the animation.
- Uses `Paused` to pause the animation at notable dates.

```python
import pandas as pd
import matplotlib.pyplot as plt
from streamjoy import stream, wrap_matplotlib, Paused

URL = "https://climexp.knmi.nl/data/ihadcrut5_global.dat"
NOTABLE_DATES = {
    "1997-12": "Kyoto Protocol adopted",
    "2005-01": "Exceeded 380 ppm",
    "2010-01": "Exceeded 390 ppm",
    "2013-05": "Exceeded 400 ppm",
    "2015-12": "Paris Agreement signed",
    "2016-01": "CO2 permanently over 400 ppm",
}


@wrap_matplotlib()
def renderer(df):
    plt.style.use("dark_background")  # Setting the style for dark mode

    fig, ax = plt.subplots()
    fig.patch.set_facecolor("#1b1e23")
    ax.set_facecolor("#1b1e23")
    ax.set_frame_on(False)
    ax.axis("off")

    # Set title
    year = df["year"].iloc[-1]
    ax.set_title(
        f"Global Temperature Anomaly {year} [HadCRUT5]",
        fontsize=15,
        loc="left",
        fontname="Courier New",
        color="lightgrey",
    )

    # draw line
    df.groupby("year")["anom"].plot(
        y="anom", color="lightgrey", legend=False, ax=ax, lw=0.5
    )

    # add source text at bottom right
    ax.text(
        0.01,
        0.05,
        f"Source: {URL}",
        va="bottom",
        ha="left",
        transform=ax.transAxes,
        fontsize=8,
        color="lightgrey",
        fontname="Courier New",
    )

    # draw end point
    jday = df.index.values[-1]
    anom = df["anom"].values[-1]
    ax.scatter(jday, anom, color="red", zorder=999)
    anom_label = f"+{anom:.1f} K" if anom > 0 else f"{anom:.1f} K"
    ax.annotate(
        anom_label,
        (jday, anom),
        textcoords="offset points",
        xytext=(-10, 5),
        fontsize=12,
        ha="right",
        va="bottom",
        color="lightgrey",
    )

    # draw yearly labels
    for year, df_year in df.reset_index().groupby("year").last().iloc[-5:].iterrows():
        if df_year["month"] != 12:
            continue
        ax.annotate(
            year,
            (df_year["jday"], df_year["anom"]),
            fontsize=12,
            ha="left",
            va="center",
            color="lightgrey",
            fontname="Courier New",
        )

    plt.subplots_adjust(bottom=0, top=0.9, left=0.05)

    month = df["date"].iloc[-1].strftime("%b")
    ax.annotate(
        month,
        (jday, anom),
        textcoords="offset points",
        xytext=(-10, 3),
        fontsize=12,
        ha="right",
        va="top",
        color="lightgrey",
        fontname="Courier New",
    )
    date_string = df["date"].iloc[-1].strftime("%Y-%m")
    if date_string in NOTABLE_DATES:
        ax.annotate(
            f"{NOTABLE_DATES[date_string]}",
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(0, -5),
            textcoords="offset points",
            fontsize=12,
            ha="left",
            va="top",
            color="lightgrey",
            fontname="Courier New",
        )
        return Paused(fig, 3)
    return fig


df = (
    pd.read_csv(
        URL,
        comment="#",
        header=None,
        sep="\s+",
        na_values=[-999.9],
    )
    .rename(columns={0: "year"})
    .melt(id_vars="year", var_name="month", value_name="anom")
)
df.index = pd.to_datetime(
    df["year"].astype(str) + df["month"].astype(str), format="%Y%m"
)
df = df.sort_index()["1995":"2024"]
df["jday"] = df.index.dayofyear
df = df.rename_axis("date").reset_index().set_index("jday")
df_list = [df[:i] for i in range(1, len(df) + 1)]

stream(df_list, renderer=renderer, threads_per_worker=1).write(
    "temperature_anomaly.mp4"
)
```