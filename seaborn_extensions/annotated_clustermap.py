from typing import Optional, List, Union, Callable
from functools import wraps

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from .types import Series, DataFrame
from .utils import minmax_scale


DEFAULT_CHANNEL_COLORS = [
    "red",
    "green",
    "blue",
    "orange",
    "purple",
    "brown",
    "pink",
    "olive",
    "cyan",
    "gray",
]

SEQUENCIAL_CMAPS = [
    "Purples",
    "Greens",
    "Oranges",
    "Greys",
    "Reds",
    "Blues",
    "YlOrBr",
    "YlOrRd",
    "OrRd",
    "PuRd",
    "RdPu",
    "BuPu",
    "GnBu",
    "PuBu",
    "YlGnBu",
    "PuBuGn",
    "BuGn",
    "YlGn",
    "binary",
    "gist_yarg",
    "gist_gray",
    "gray",
    "bone",
    "pink",
    "spring",
    "summer",
    "autumn",
    "winter",
    "cool",
    "Wistia",
    "hot",
    "afmhot",
    "gist_heat",
    "copper",
]


def is_numeric(x: Series) -> bool:
    if (
        x.dtype.name
        in [
            "float",
            "float32",
            "float64",
            "int",
            "int32",
            "int64",
            "Int64",
        ]
        or is_datetime(x)
    ):
        return True
    if x.dtype.name in ["object", "string", "boolean", "bool", "category"]:
        return False
    raise ValueError(f"Cannot transfer data type '{x.dtype}' to color!")


def is_datetime(x: Series) -> bool:
    if "datetime" in x.dtype.name:
        return True
    return False


def to_numeric(x: Series) -> Series:
    """Encode a string or categorical series to integer type."""
    res = pd.Series(
        index=x.index, dtype=float
    )  # this will imply np.nan keeps being np.nan
    for i, v in enumerate(x.value_counts().sort_index().index):
        res.loc[x == v] = i
    return res


def get_categorical_cmap(x: Series) -> matplotlib.colors.ListedColormap:
    """Choose a colormap for a categorical series encoded as ints."""
    # TODO: allow choosing from sets of categorical cmaps.
    # additional ones could be Pastel1/2, Set2/3

    # colormaps are truncated to existing values
    n = int(x.max() + 1)
    for v in [10, 20]:
        if n < v:
            return matplotlib.colors.ListedColormap(
                colors=plt.get_cmap(f"tab{v}").colors[:n], name=f"tab{v}-{n}"
            )
    if n < 40:
        return matplotlib.colors.ListedColormap(
            colors=np.concatenate(
                [
                    plt.get_cmap("tab20c")(range(20)),
                    plt.get_cmap("tab20b")(range(20)),
                ]
            )[:n],
            name=f"tab40-{n}",
        )
    raise ValueError("Only up to 40 unique values can be plotted as color.")


def to_color_series(x: Series, cmap: Optional[str] = "Greens") -> Series:
    """
    Map a numeric pandas series to a series of RBG values.
    NaN values are white.
    """
    if is_numeric(x):
        return pd.Series(
            plt.get_cmap(cmap)(minmax_scale(x)).tolist(),
            index=x.index,
            name=x.name,
        )
    # str or categorical
    res = to_numeric(x)
    _cmap = get_categorical_cmap(res)
    # float values passed to cmap must be in [0.0-1.0] range
    return pd.Series(
        _cmap(res / res.max()).tolist(), index=x.index, name=x.name
    )


def to_color_dataframe(
    x: Union[Series, DataFrame],
    cmaps: Optional[Union[str, List[str]]] = None,
    offset: int = 0,
) -> DataFrame:
    """Map a numeric pandas DataFrame to RGB values."""
    if isinstance(x, pd.Series):
        x = x.to_frame()
    if cmaps is None:
        # the offset is in order to get different colors for rows and columns by default
        cmaps = [plt.get_cmap(cmap) for cmap in SEQUENCIAL_CMAPS[offset:]]
    if isinstance(cmaps, str):
        cmaps = [cmaps]
    return pd.concat(
        [to_color_series(x[col], cmap) for col, cmap in zip(x, cmaps)], axis=1
    )


def _add_extra_colorbars_to_clustermap(
    grid: sns.matrix.ClusterGrid,
    datas: Union[Series, DataFrame],
    cmaps: Optional[Union[str, List[str]]] = None,
    # location: Union[Literal["col"], Literal["row"]] = "row",
    location: str = "row",
) -> None:
    """Add either a row or column colorbar to a seaborn Grid."""

    def add(
        data: Series, cmap: str, bbox: List[List[int]], orientation: str
    ) -> None:
        ax = grid.fig.add_axes(matplotlib.transforms.Bbox(bbox))
        if is_numeric(data):
            if is_datetime(data):
                data = minmax_scale(data)
            norm = matplotlib.colors.Normalize(vmin=data.min(), vmax=data.max())
            cbar = matplotlib.colorbar.ColorbarBase(
                ax,
                cmap=plt.get_cmap(cmap),
                norm=norm,
                orientation=orientation,
                label=data.name,
            )
        else:
            res = to_numeric(data)
            # res /= res.max()
            cmap = get_categorical_cmap(res)
            # norm = matplotlib.colors.Normalize(vmin=res.min(), vmax=res.max())
            cbar = matplotlib.colorbar.ColorbarBase(
                ax,
                cmap=cmap,
                orientation=orientation,
                label=data.name,
            )
            cbar.set_ticks(res.drop_duplicates().sort_values() / res.max())
            cbar.set_ticklabels(data.value_counts().sort_index().index)

    offset = 1 if location == "row" else 0

    if isinstance(datas, pd.Series):
        datas = datas.to_frame()
    if cmaps is None:
        cmaps = SEQUENCIAL_CMAPS[offset:]
    if isinstance(cmaps, str):
        cmaps = [cmaps]

    # get position to add new axis in existing figure
    # # get_position() returns ((x0, y0), (x1, y1))
    heat = grid.ax_heatmap.get_position()
    cbar_spacing = 0.05
    cbar_size = 0.025
    if location == "col":
        orientation = "vertical"
        dend = grid.ax_col_dendrogram.get_position()
        y0 = dend.y0
        y1 = dend.y1
        for i, (data, cmap) in enumerate(zip(datas, cmaps)):
            if i == 0:
                x0 = heat.x1
                x1 = heat.x1 + cbar_size
            else:
                x0 += cbar_size + cbar_spacing
                x1 += cbar_size + cbar_spacing
            add(datas[data], cmap, [[x0, y0], [x1, y1]], orientation)
    else:
        orientation = "horizontal"
        dend = grid.ax_row_dendrogram.get_position()
        x0 = dend.x0
        x1 = dend.x1
        for i, (data, cmap) in enumerate(zip(datas, cmaps)):
            if i == 0:
                y0 = dend.y0 - cbar_size
                y1 = dend.y0
            else:
                y0 -= cbar_size + cbar_spacing
                y1 -= cbar_size + cbar_spacing
            add(datas[data], cmap, [[x0, y0], [x1, y1]], orientation)


def _add_colorbars(
    grid: sns.matrix.ClusterGrid,
    rows: DataFrame = None,
    cols: DataFrame = None,
    row_cmaps: Optional[List[str]] = None,
    col_cmaps: Optional[List[str]] = None,
) -> None:
    """Add row and column colorbars to a seaborn Grid."""
    if rows is not None:
        _add_extra_colorbars_to_clustermap(
            grid, rows, location="row", cmaps=row_cmaps
        )
    if cols is not None:
        _add_extra_colorbars_to_clustermap(
            grid, cols, location="col", cmaps=col_cmaps
        )


def colorbar_decorator(f: Callable) -> Callable:
    """
    Decorate seaborn.clustermap in order to have numeric values passed to the
    ``row_colors`` and ``col_colors`` arguments translated into row and column
    annotations and in addition colorbars for the restpective values.
    """

    def clustermap(*args, **kwargs):
        # Defaults

        # # Size of figure
        if "figsize" not in kwargs:
            kwargs["figsize"] = (10, 10)
        if kwargs["figsize"] == (10, 10):  # default value
            # assumes pivot_kws is not used...
            # would depend on x/y-ticklabel size...
            ...

        # # Decide if labeling x/y-ticklabels based on shape
        max_items = 200
        data = args[0]
        if "xticklabels" not in kwargs:
            kwargs["xticklabels"] = True if data.shape[0] < max_items else False
        if "yticklabels" not in kwargs:
            kwargs["yticklabels"] = True if data.shape[1] < max_items else False

        # dendrogram aspect ratio
        d = 0.1
        aspect = kwargs["figsize"][0] / kwargs["figsize"][1]
        smallest = (
            np.argmin(kwargs["figsize"])
            if len(np.unique(kwargs["figsize"])) > 1
            else -1
        )
        if smallest == -1:
            s = 1
            dar = (d, d)
        else:
            s = kwargs["figsize"][smallest] * d
            dar = tuple(
                [
                    d if i == smallest else s / kwargs["figsize"][i]
                    for i in range(2)
                ]
            )

        if "cbar_kws" not in kwargs:
            kwargs["cbar_kws"] = dict()

        if smallest == 0:
            kwargs["cbar_kws"].update(dict(aspect=20 / aspect))

        # # non-Z-score mode:
        nz_default_kws = dict(dendrogram_ratio=dar, metric="correlation")
        # # Z-score mode:
        zs_default_kws = dict(
            z_score=1,
            center=0,
            cmap="RdBu_r",
            robust=True,
            dendrogram_ratio=dar,
            metric="correlation",
        )
        if "config" in kwargs:
            default_kws = (
                zs_default_kws
                if kwargs["config"].lower() in ["z", "z_score", "z-score"]
                else nz_default_kws
            )
            kwargs.update(default_kws)
            del kwargs["config"]

        # Annotations:
        cmaps = {"row": None, "col": None}
        # # capture "row_cmaps" and "col_cmaps" out of the kwargs
        for arg in ["row", "col"]:
            if arg + "_colors_cmaps" in kwargs:
                # TODO: make sure this matches in type/length the row/col_colors kwargs.
                cmaps[arg] = kwargs[arg + "_colors_cmaps"]
                del kwargs[arg + "_colors_cmaps"]

        # # get dataframe with colors and respective colormaps for rows and cols
        # # instead of the original numerical values
        _kwargs = dict(rows=None, cols=None)
        for arg in ["row", "col"]:
            if arg + "_colors" in kwargs:
                if isinstance(
                    kwargs[arg + "_colors"], (pd.DataFrame, pd.Series)
                ):
                    _kwargs[arg + "s"] = kwargs[arg + "_colors"]
                    kwargs[arg + "_colors"] = to_color_dataframe(
                        x=kwargs[arg + "_colors"],
                        cmaps=cmaps[arg],
                        offset=1 if arg == "row" else 0,
                    )

        # Call original function
        grid = f(*args, **kwargs)

        # Add the colorbar legends to the figure
        _add_colorbars(
            grid, **_kwargs, row_cmaps=cmaps["row"], col_cmaps=cmaps["col"]
        )
        # Some nicities
        if grid.ax_heatmap.get_xlabel() in ["", None]:
            grid.ax_heatmap.set_xlabel(f"(n = {data.shape[0]})")
        if grid.ax_heatmap.get_ylabel() in ["", None]:
            grid.ax_heatmap.set_ylabel(f"(n = {data.shape[1]})")
        return grid

    # TODO: edit original seaborn.clustermap docstring to document {row,col}_colors_cmaps arguments.
    start_docs = "{row,col}_colors : "
    end_docs = "mask : boolean"
    add_docs = """{row,col}_colors : list-like or pandas DataFrame/Series, optional
            List of colors to label for either the rows or columns. Useful to
            evaluate whether samples within a group are clustered together. Can
            use nested lists or DataFrame for multiple color levels of labeling.
            If given as a DataFrame or Series, labels for the colors are extracted
            from the DataFrames column names or from the name of the Series.
            DataFrame/Series colors are also matched to the data by their
            index, ensuring colors are drawn in the correct order.

            TODO: complete defining new behavious
        {row,col}_colors_cmaps:
            TODO: describe
        """

    docs = sns.clustermap.__doc__
    a = docs.index(start_docs)
    o = docs.index(end_docs)

    clustermap.__doc__ = docs[:a] + add_docs + docs[o:]

    # Add a flag
    f.decorated = True

    return clustermap


def activate():
    if sns.clustermap.__module__ != "seaborn_extensions.annotated_clustermap":
        sns.clustermap = colorbar_decorator(sns.clustermap)
