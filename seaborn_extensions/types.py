from typing import Union
from collections.abc import MutableSequence

import pandas
import numpy
import matplotlib

# type aliasing (done with Union to distinguish from other declared variables)
Array = Union[numpy.ndarray]
Series = Union[pandas.Series]
MultiIndexSeries = Union[pandas.Series]
DataFrame = Union[pandas.DataFrame]

Figure = Union[matplotlib.figure.Figure]
Axis = Union[matplotlib.axis.Axis]
Patch = Union[matplotlib.patches.Patch]
ColorMap = Union[matplotlib.colors.LinearSegmentedColormap]


Iterables = Union[set, MutableSequence, pandas.Series, pandas.Index]
