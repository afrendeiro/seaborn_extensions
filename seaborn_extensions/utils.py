"""
Utility functions used throughout the package.
"""

import typing as tp
from functools import wraps
from inspect import signature

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from seaborn_extensions import SEQUENCIAL_CMAPS
from seaborn_extensions.types import Array, Series, DataFrame


def is_documented_by(original):
    def wrapper(target):
        target.__doc__ = original.__doc__
        return target

    return wrapper


def filter_kwargs_by_callable(
    kwargs: tp.Dict[str, tp.Any],
    callabl: tp.Callable,
    exclude: tp.List[str] = None,
    allow_kwargs: bool = False,
) -> tp.Dict[str, tp.Any]:
    """Filter a dictionary keeping only the keys which are part of a function signature."""
    args = signature(callabl).parameters.keys()
    if allow_kwargs and ("kwargs" in args):
        return kwargs
    return {k: v for k, v in kwargs.items() if (k in args) and k not in (exclude or [])}


@tp.overload
def minmax_scale(x: Array) -> Array:
    ...


@tp.overload
def minmax_scale(x: DataFrame) -> DataFrame:
    ...


def minmax_scale(x: tp.Union[Array, DataFrame]) -> tp.Union[Array, DataFrame]:
    with np.errstate(divide="ignore", invalid="ignore"):
        return (x - x.min()) / (x.max() - x.min())


def get_grid_dims(
    dims: tp.Union[int, tp.Collection], _nstart: int = None
) -> tp.Tuple[int, int]:
    """
    Given a number of `dims` subplots, choose optimal x/y dimentions of plotting
    grid maximizing in order to be as square as posible and if not with more
    columns than rows.
    """
    if not isinstance(dims, int):
        dims = len(dims)
    if _nstart is None:
        n = min(dims, 1 + int(np.ceil(np.sqrt(dims))))
    else:
        n = _nstart
    if (n * n) == dims:
        m = n
    else:
        a = pd.Series(n * np.arange(1, n + 1)) / dims
        m = a[a >= 1].index[0] + 1
    assert n * m >= dims

    if n * m % dims > 1:
        try:
            n, m = get_grid_dims(dims=dims, _nstart=n - 1)
        except IndexError:
            pass
    return n, m


def close_plots(func: tp.Callable) -> None:
    """
    Decorator to close all plots on function exit.
    """

    @wraps(func)
    def close(*args, **kwargs):
        func(*args, **kwargs)
        plt.close("all")

    return close


def is_numeric(x: tp.Union[Series, tp.Any]) -> bool:
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    if (
        x.dtype.name
        in [
            "float",
            "float32",
            "float64",
            "int",
            "int8",
            "int16",
            "int32",
            "int64",
            "Int64",
        ]
        or is_datetime(x)
    ):
        return True
    if x.dtype.name in ["object", "string", "boolean", "bool"]:
        return False
    if x.dtype.name == "category":
        if len(set(type(i) for i in x)) != 1:
            raise ValueError("Series contains mixed types. Cannot transfer to color!")
        return is_numeric(x.iloc[0])
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


def log_pvalues(x, f: float = 0.1):
    """
    Calculate -log10(p-value) of array.

    Replaces infinite values with:

    .. highlight:: python
    .. code-block:: python

        max(x) + max(x) * f

    that is, fraction ``f`` more than the maximum non-infinite -log10(p-value).

    Parameters
    ----------
    x : :class:`pandas.Series`
        Series with numeric values
    f : :obj:`float`
        Fraction to augment the maximum value by if ``x`` contains infinite values.

        Defaults to 0.1.

    Returns
    -------
    :class:`pandas.Series`
        Transformed values.
    """
    ll = -np.log10(x)
    rmax = ll[ll != np.inf].max()
    return ll.replace(np.inf, rmax + rmax * f)


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


def to_color_series(x: Series, cmap: tp.Optional[str] = None) -> Series:
    """
    Map a numeric pandas series to a series of RBG values.
    NaN values are white.
    """
    cmap_types = (
        matplotlib.colors.ListedColormap,
        matplotlib.colors.LinearSegmentedColormap,
    )

    if is_numeric(x):
        if cmap is None:
            cmap = "Greens"
        return pd.Series(
            plt.get_cmap(cmap)(minmax_scale(x.astype(float))).tolist(),
            index=x.index,
            name=x.name,
        )
    # str or categorical
    res = to_numeric(x)
    if cmap is None or isinstance(
        cmap, cmap_types[1]
    ):  # matching a LinearSegmentedColormap means it was probably passed a default
        _cmap = get_categorical_cmap(res)
    elif isinstance(cmap, str):
        _cmap = plt.get_cmap(cmap)
    elif isinstance(cmap, cmap_types[0]):
        _cmap = cmap
    elif isinstance(cmap, (list, np.ndarray)):
        _cmap = matplotlib.colors.ListedColormap(cmap, name="custom")
    elif isinstance(cmap, float):
        raise ValueError(
            "Please provide same number of `row/col_colors_cmaps` as `row/col_colors`."
        )
    else:
        raise ValueError(
            f"Could not understand values passed as `row/col_colors_cmaps`: {cmap}."
        )

    # float values passed to cmap must be in [0.0-1.0] range
    return pd.Series(_cmap(res / res.max()).tolist(), index=x.index, name=x.name)


def to_color_dataframe(
    x: tp.Union[Series, DataFrame],
    cmaps: tp.Optional[tp.Union[str, tp.Sequence[str]]] = None,
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
