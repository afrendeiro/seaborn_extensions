from typing import Union

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
