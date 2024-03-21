# Gender gapminder

<video controls="true" allowfullscreen="true">
<source src="https://github.com/ahuang11/streamjoy/assets/15331990/a684665b-e049-490c-8459-d6ae274160bf" type="video/mp4">
</video>

This example demonstrates how to use `stream` and `connect` to create a video
comparing gender population data from the Gapminder dataset.

Highlights:

- Uses `intro_title` and `intro_subtitle` to set the title and subtitle of the video.
- Uses `renderer_kwargs` to pass keyword arguments to the custom `renderer` function.
- Updates `fps` to 30 to create a smoother animation.
- Uses `connect` to concatenate the two heterogeneous streams (different keyword arguments with different titles) together.

```python
import pandas as pd
from streamjoy import stream, connect

if __name__ == "__main__":
    url_fmt = (
        "https://raw.githubusercontent.com/open-numbers/ddf--gapminder--systema_globalis/master/"
        "countries-etc-datapoints/ddf--datapoints--{gender}_population_with_projections--by--geo--time.csv"
    )
    df = pd.concat((
        pd.read_csv(url_fmt.format(gender=gender)).set_index(["geo", "time"])
        for gender in ["male", "female"]),
        axis=1,
    )

    streams = []
    for country in ["usa", "chn"]:
        df_sub = df.loc[country].reset_index().melt("time")
        streamed = stream(
            df_sub,
            groupby="variable",
            intro_title="Gapminder",
            intro_subtitle=f"{country.upper()} Male vs Female Population",
            renderer_kwargs={
                "x": "time",
                "y": "value",
                "xlabel": "Year",
                "ylabel": "Population",
                "title": f"{country.upper()} {{time}}",
            },
            max_frames=-1,
            fps=30,
        )
        streams.append(streamed)
    connect(streams).write("gender_gapminder.mp4")
```