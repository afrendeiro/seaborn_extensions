from typing import Union, Optional, Tuple, Collection, overload

import numpy as np
import pandas as pd

from .types import Array, DataFrame


@overload
def minmax_scale(x: Array) -> Array:
    ...


@overload
def minmax_scale(x: DataFrame) -> DataFrame:
    ...


def minmax_scale(x: Union[Array, DataFrame]) -> Union[Array, DataFrame]:
    with np.errstate(divide="ignore", invalid="ignore"):
        return (x - x.min()) / (x.max() - x.min())


def get_grid_dims(
    dims: Union[int, Collection], _nstart: Optional[int] = None
) -> Tuple[int, int]:
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
