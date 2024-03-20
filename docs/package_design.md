# Package design

## 🪪 Naming

StreamJoy stems from the idea of streaming parallelized, output images to GIF or MP4, *as they get serialized*.

This was a mini breakthrough for me, as I had written other packages to try efficiently animating data (e.g. [`enjoyn`](https://enjoyn.readthedocs.io/en/latest/) and [`ahlive`](https://ahlive.readthedocs.io/en/latest/)). However, both of these packages suffered from the bottleneck of having to wait for all the images to get written out to disk before starting generating the animation.

After, discovering this breakthrough, it brought me joy, and I wanted to share that joy with others by writing a package that reduces the boilerplate and time to work on animations, bringing joy to the user.

Coincidentally, SJ is also my wife's initials, so it was a perfect fit! :D

I also was thinking of naming this `streamio` and `streamit`, but the former was already taken and the latter too close to `streamlit`.

## 📶 Diagram

Below is a diagram of the package design. The animation part is actually quite simple--most of the complexity comes with handling various input types, e.g. URLs, files, and datasets.

<figure>
    <img src="https://github.com/ahuang11/streamjoy/raw/main/docs/assets/design.svg" alt="StreamJoy package design" style="width:100%">
    <figcaption>StreamJoy package design</figcaption>
</figure>

```mermaid
graph TD
    A[Start] --> Z{Input Type}
    Z -->|URL| U[Download and Assess Content]
    Z -->|Direct Input| V{Determine Content Type}
    U --> V
    V -->|DataFrame| B[Split pandas DataFrame]
    V -->|XArray Dataset/DataArray| X[Split XArray by dim]
    V -->|HoloViews HoloMap/DynamicMap| H[Split HoloViews by kdim]
    V -->|Directory of Images| Y[Glob all files]
    V -->|List of Images| D[Open MP4/GIF buffer and start Dask client]
    B --> D
    X --> D
    H --> D
    Y --> D
    D --> E{Process each frame in parallel with Dask}
    E -->|Renderer available| F[Call custom/default renderer]
    E -->|Renderer not available| G[Convert to np.array with imread]
    F --> G
    G --> I[Stream to MP4/GIF buffer with imwrite]
    I --> J{All units processed?}
    J -->|Yes| K[Save MP4/GIF]
    J -->|No| E
    K -->|Optimize GIF?| N[Optimize GIF]
    N --> L[End]
    K --> L
```