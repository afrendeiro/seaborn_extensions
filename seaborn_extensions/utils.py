from typing import Union, overload

import numpy as np

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
