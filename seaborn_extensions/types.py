import typing as tp
from collections.abc import MutableSequence

import pandas
import numpy
import matplotlib

# type aliasing (done with Union to distinguish from other declared variables)
Array = numpy.ndarray
Series = pandas.Series
MultiIndexSeries = pandas.Series
DataFrame = pandas.DataFrame

Figure = matplotlib.figure.Figure
Axis = matplotlib.axis.Axis
Patch = matplotlib.patches.Patch
ColorMap = matplotlib.colors.LinearSegmentedColormap

Iterables = tp.Union[set, MutableSequence, pandas.Series, pandas.Index]
