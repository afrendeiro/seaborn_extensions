"""
A replacement of seaborn.clustermap with additional features.
"""


import typing as tp
from typing import Optional, List, Union, Callable
import warnings

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from seaborn_extensions import SEQUENCIAL_CMAPS
from seaborn_extensions.types import Series, DataFrame, Array, Figure
from seaborn_extensions.utils import (
    minmax_scale,
    is_numeric,
    is_datetime,
    to_numeric,
    get_categorical_cmap,
    to_color_dataframe,
)


# TODO: revamp supporting custom cmaps/palettes


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
    max_items = 120
    data = args[0]
    if "xticklabels" not in kwargs:
        kwargs["xticklabels"] = data.shape[1] < max_items
    if "yticklabels" not in kwargs:
        kwargs["yticklabels"] = data.shape[0] < max_items

    # dendrogram aspect ratio
    d = 0.1
    aspect = kwargs["figsize"][0] / kwargs["figsize"][1]
    smallest = (
        np.argmin(kwargs["figsize"]) if len(np.unique(kwargs["figsize"])) > 1 else -1
    )
    if smallest == -1:
        s = 1
        dar = (d, d)
    else:
        s = kwargs["figsize"][smallest] * d
        dar = tuple(d if i == smallest else s / kwargs["figsize"][i] for i in range(2))

    if "cbar_kws" not in kwargs:
        kwargs["cbar_kws"] = dict()

    if smallest == 0:
        kwargs["cbar_kws"].update(dict(aspect=20 / aspect))

    # # non-Z-score mode:
    nz_default_kws = dict(
        cmap="Reds", robust=True, dendrogram_ratio=dar, metric="correlation"
    )
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
            if kwargs["config"].lower() in ["z", "zscore", "z_score", "z-score"]
            else nz_default_kws
        )
        # kwargs.update(default_kws)  # for overwrite
        for k, v in default_kws.items():
            if k not in kwargs:
                kwargs[k] = v
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
            if isinstance(kwargs[arg + "_colors"], (pd.DataFrame, pd.Series)):
                _kwargs[arg + "s"] = kwargs[arg + "_colors"]
                kwargs[arg + "_colors"] = to_color_dataframe(
                    x=kwargs[arg + "_colors"],
                    cmaps=cmaps[arg],
                    offset=1 if arg == "row" else 0,
                )

    # Add p-value annotation
    # TODO: document
    if "pvalues" in kwargs:
        assert "annot" not in kwargs, "If providing p-values, `annot` cannot be used!"
        p = kwargs["pvalues"]
        # TODO: allow custom thresholds
        p = ((p < 0.05) & (p > 0.01)).replace({True: 1}) + ((p < 0.01)).replace({True: 2})
        kwargs["annot"] = p

    # Call original function
    grid = sns.clustermap(*args, **kwargs)

    # Add the colorbar legends to the figure
    _add_colorbars(grid, **_kwargs, row_cmaps=cmaps["row"], col_cmaps=cmaps["col"])
    # Some niceties
    # if grid.ax_heatmap.get_xlabel() in ["", None]:
    ax = grid.ax_heatmap
    ax.set_xlabel(f"{ax.get_xlabel()}\n(n = {data.shape[1]})")
    # if ax.get_ylabel() in ["", None]:
    ax.set_ylabel(f"{ax.get_ylabel()}\n(n = {data.shape[0]})")

    # Convert numeric p-value annotation to text
    if "pvalues" in kwargs:
        r = {"0": "", "1": "*", "2": "**"}
        for c in ax.get_children():
            if isinstance(c, matplotlib.text.Text):
                if c.get_text() in r:
                    c.set_text(r[c.get_text()])

    return grid


def _add_extra_colorbars_to_clustermap(
    grid: sns.matrix.ClusterGrid,
    datas: Union[Series, DataFrame],
    cmaps: Optional[Union[str, List[str]]] = None,
    # location: Union[Literal["col"], Literal["row"]] = "row",
    location: str = "row",
) -> None:
    """Add either a row or column colorbar to a seaborn Grid."""

    def add(data: Series, cmap: str, bbox: List[List[int]], orientation: str) -> None:
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
        _add_extra_colorbars_to_clustermap(grid, rows, location="row", cmaps=row_cmaps)
    if cols is not None:
        _add_extra_colorbars_to_clustermap(grid, cols, location="col", cmaps=col_cmaps)


def _add_docs_to_clustermap():
    """
    Edit original seaborn.clustermap docstring to document {row,col}_colors_cmaps arguments.
    """
    # TODO: finish documenting changes.
    error_msg = (
        "Seaborn version may not be compatible with seaborn_extensions version."
        "Skipping annotating clustermap function docstring."
    )

    docs = sns.clustermap.__doc__
    anchors = np.asarray(
        [("pivot_kws : ", "method : "), ("{row,col}_colors : ", "mask : bool")]
    )

    points = np.zeros(anchors.shape, dtype=int)
    for i, tup in enumerate(anchors):
        for j, p in enumerate(tup):
            try:
                x = docs.index(p)
            except ValueError:
                print(error_msg)
                return
            points[i, j] = x

    add_docs1 = """config : str, optional
            EXTENSION!
            One of two pre-defined configurations: "abs", "zscore".
            These two configurations provide custom default keyword arguments
            compared with the native seaborn function and several adjustments to
            figure and axis sizes, labels and other objects.
            Options:
             - "abs": good for non-negative data.
             - "zscore": good for real data with variables with very different means.
            Other keyword arguments affected (only is not provided):
             - {x,y}ticklabels: will turn off if more than 120 items in each axis.
             - dendrogram_ratio: will adjust, given relative shape of data.
        """
    add_docs2 = """{row,col}_colors : list-like or pandas DataFrame/Series, optional
            EXTENSION!
            List of colors to label for either the rows or columns. Useful to
            evaluate whether samples within a group are clustered together. Can
            use nested lists or DataFrame for multiple color levels of labeling.
            If given as a DataFrame or Series, labels for the colors are extracted
            from the DataFrames column names or from the name of the Series.
            DataFrame/Series colors are also matched to the data by their
            index, ensuring colors are drawn in the correct order.

            TODO: complete defining new behavious
        {row,col}_colors_cmaps:
            EXTENSION!
            TODO: describe
        """
    clustermap.__doc__ = (
        docs[: points[0][0]]
        + add_docs1
        + docs[points[0][1] : points[1][0]]
        + add_docs2
        + docs[points[1][1] :]
    )


_add_docs_to_clustermap()


def colorbar_decorator(f: Callable) -> Callable:
    """
    Decorate seaborn.clustermap in order to have numeric values passed to the
    ``row_colors`` and ``col_colors`` arguments translated into row and column
    annotations and in addition colorbars for the restpective values.
    """
    # Add a flag
    f.decorated = True

    return clustermap


def activate():
    warnings.warn(
        "Decoration of native searborn.clustermap function will be deprecated in version 1.0.0, use 'from seaborn_extensions import clustermap' instead.",
        PendingDeprecationWarning,
    )
    if sns.clustermap.__module__ != "seaborn_extensions.annotated_clustermap":
        sns.clustermap = colorbar_decorator(sns.clustermap)


# To plot just the attribute heatmap:
def get_attribute_colors(
    y: DataFrame,
    attributes: tp.Sequence[str],
    palettes: tp.Mapping[str, tp.Tuple[float]],
    cmaps: tp.Mapping[str, str],
    as_dataframe: bool = False,
) -> tp.Union[Array, DataFrame]:
    vals = list()
    for attr in attributes:
        if attr in palettes:
            p = dict(zip(y[attr].cat.categories, palettes[attr]))
            val = np.asarray([p[v] if not pd.isnull(v) else (0, 0, 0) for v in y[attr]])
        elif attr in cmaps:
            cmap = plt.get_cmap(cmaps[attr])
            val = cmap(minmax_scale(y[attr].astype(float)))[:, :3]
        vals.append(val)
    if as_dataframe:
        return pd.DataFrame(
            map(tuple, np.asarray(vals)), index=attributes, columns=y.index
        )
    return np.asarray(vals)


def plot_attribute_heatmap(
    y: DataFrame,
    attributes: tp.Sequence[str],
    palettes: tp.Mapping[str, tp.Tuple[float]],
    cmaps: tp.Mapping[str, str],
    **kwargs,
) -> Figure:
    vals = get_attribute_colors(y, attributes, palettes, cmaps)
    if "ax" not in kwargs:
        fig, axes = plt.subplots(
            len(attributes), **kwargs, gridspec_kw=dict(wspace=0, hspace=0)
        )
    else:
        fig = kwargs["ax"].figure
    # ax.imshow(vals)
    for _p, attr, ax in zip(vals, attributes, axes):
        ax.imshow(_p[np.newaxis, ...])
        ax.set(xticks=[], yticks=[0])
        ax.set_yticklabels([attr], rotation=0)
        sns.despine(ax=ax, left=True, bottom=True)
    ax = axes[-1]
    ax.set_xticks(range(len(y.index)))
    ax.set_xticklabels(y.index, rotation=90)
    return fig
