# CO2 timeseries

<video controls="true" allowfullscreen="true">
<source src="https://github.com/ahuang11/streamjoy/assets/15331990/1f6fa5ae-9298-452d-ae1c-41d8c9f6cd34" type="video/mp4">
</video>

Shows the yearly CO2 measurements from the Mauna Loa Observatory in Hawaii.

The data is sourced from the [datasets/co2-ppm-daily](https://github.com/datasets/co2-ppm-daily/blob/master/co2-ppm-daily-flow.py).

Highlights:

- Uses `wrap_matplotlib` to automatically handle saving and closing the figure.
- Uses a custom `renderer` function to create each frame of the animation.
- Uses `Paused` to pause the animation at notable dates.

```python
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from streamjoy import stream, wrap_matplotlib, Paused

URL = "https://raw.githubusercontent.com/datasets/co2-ppm-daily/master/data/co2-ppm-daily.csv"
NOTABLE_YEARS = {
    1958: "Mauna Loa measurements begin",
    1979: "1st World Climate Conference",
    1997: "Kyoto Protocol",
    2005: "exceeded 380 ppm",
    2010: "exceeded 390 ppm",
    2013: "exceeded 400 ppm",
    2015: "Paris Agreement",
}


@wrap_matplotlib()
def renderer(df):
    plt.style.use("dark_background")

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#1b1e23")
    ax.set_facecolor("#1b1e23")
    ax.set_frame_on(False)
    ax.axis("off")
    ax.set_title(
        "CO2 Yearly Max",
        fontsize=20,
        loc="left",
        fontname="Courier New",
        color="lightgrey",
    )

    # draw line
    df.plot(
        y="value",
        color="lightgrey",  # Line color
        legend=False,
        ax=ax,
    )

    # max date
    max_date = df["value"].idxmax()
    max_co2 = df["value"].max()
    ax.text(
        0.0,
        0.92,
        f"{max_co2:.0f} ppm",
        va="bottom",
        ha="left",
        transform=ax.transAxes,
        fontsize=25,
        color="lightgrey",
    )
    ax.text(
        0.0,
        0.91,
        f"Peaked in {max_date.year}",
        va="top",
        ha="left",
        transform=ax.transAxes,
        fontsize=12,
        color="lightgrey",
        fontname="Courier New",
    )

    # draw end point
    date = df.index[-1]
    co2 = df["value"].values[-1]
    diff = df["diff"].fillna(0).values[-1]
    diff = f"+{diff:.0f}" if diff >= 0 else f"{diff:.0f}"
    ax.scatter(date, co2, color="red", zorder=999)
    ax.annotate(
        f"{diff} ppm",
        (date, co2),
        textcoords="offset points",
        xytext=(-10, 5),
        fontsize=12,
        ha="right",
        va="bottom",
        color="lightgrey",
    )

    # draw source label
    ax.text(
        0.0,
        0.03,
        f"Source: {URL}",
        va="bottom",
        ha="left",
        transform=ax.transAxes,
        fontsize=8,
        color="lightgrey",
    )

    # properly tighten layout
    plt.subplots_adjust(bottom=0, top=0.9, right=0.9, left=0.05)

    # pause at notable years
    year = date.year
    if year in NOTABLE_YEARS:
        ax.annotate(
            f"{NOTABLE_YEARS[year]} - {year}",
            (date, co2),
            textcoords="offset points",
            xytext=(-10, 3),
            fontsize=10.5,
            ha="right",
            va="top",
            color="lightgrey",
            fontname="Courier New",
        )
        return Paused(ax, 2.8)
    else:
        ax.annotate(
            year,
            (date, co2),
            textcoords="offset points",
            xytext=(-10, 3),
            fontsize=10.5,
            ha="right",
            va="top",
            color="lightgrey",
            fontname="Courier New",
        )
        return ax


if __name__ == "__main__":
    df = (
        pd.read_csv(URL, parse_dates=True, index_col="date")
        .resample("1YE")
        .max()
        .interpolate()
        .assign(
            diff=lambda df: df["value"].diff(),
        )
    )
    stream(df, renderer=renderer, max_frames=-1, threads_per_worker=1).write("co2_emissions.mp4")
```