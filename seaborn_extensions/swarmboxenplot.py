"""Main module."""


from typing import Any, Tuple, Union, Dict, Literal, Optional, overload
import itertools

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg


from seaborn_extensions.types import DataFrame, Axis, Figure


def add_transparency_to_boxenplot(ax: Axis) -> None:
    patches = (
        matplotlib.collections.PatchCollection,
        matplotlib.collections.PathCollection,
    )
    [x.set_alpha(0.25) for x in ax.get_children() if isinstance(x, patches)]


@overload
def swarmboxenplot(
    data: DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    swarm: bool = True,
    boxen: bool = True,
    ax: None = None,
    test: Literal[False] = False,
) -> Figure:
    ...


@overload
def swarmboxenplot(
    data: DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    swarm: bool = True,
    boxen: bool = True,
    ax: Axis = Axis,
    test: Literal[False] = False,
) -> None:
    ...


@overload
def swarmboxenplot(
    data: DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    swarm: bool = True,
    boxen: bool = True,
    ax: None = None,
    test: Literal[True] = True,
) -> Tuple[Figure, DataFrame]:
    ...


@overload
def swarmboxenplot(
    data: DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    swarm: bool = True,
    boxen: bool = True,
    ax: Axis = Axis,
    test: Literal[True] = True,
) -> DataFrame:
    ...


def swarmboxenplot(
    data: DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    swarm: bool = True,
    boxen: bool = True,
    ax: Optional[Axis] = None,
    test: bool = True,
    multiple_testing: Union[bool, str] = "fdr_bh",
    test_upper_threshold: float = 0.05,
    test_lower_threshold: float = 0.01,
    plot_non_significant: bool = False,
    plot_kws: Optional[Dict[str, Any]] = None,
    test_kws: Optional[Dict[str, Any]] = None,
) -> Optional[Union[Figure, DataFrame, Tuple[Figure, DataFrame]]]:
    """
    # Testing:

    data = pd.DataFrame(
        [np.random.random(20), np.random.choice(['a', 'b'], 20)],
        index=['cont', 'cat']).T.convert_dtypes()
    data.loc[data['cat'] == 'b', 'cont'] *= 5
    fig = swarmboxenplot(data=data, x='cat', y='cont')


    data = pd.DataFrame(
        [np.random.random(40), np.random.choice(['a', 'b', 'c'], 40)],
        index=['cont', 'cat']).T.convert_dtypes()
    data.loc[data['cat'] == 'b', 'cont'] *= 5
    data.loc[data['cat'] == 'c', 'cont'] -= 5
    fig = swarmboxenplot(data=data, x='cat', y='cont', test_kws=dict(parametric=True))

    """
    if test_kws is None:
        test_kws = dict()
    if plot_kws is None:
        plot_kws = dict()

    if ax is None:
        fig, _ax = plt.subplots(1, 1, figsize=(4, 4))
    else:
        _ax = ax
    if boxen:
        sns.boxenplot(data=data, x=x, y=y, hue=hue, ax=_ax, **plot_kws)
    if boxen and swarm:
        add_transparency_to_boxenplot(_ax)
    if swarm:
        sns.swarmplot(data=data, x=x, y=y, hue=hue, ax=_ax, **plot_kws)
    _ax.set_xticklabels(_ax.get_xticklabels(), rotation=90)

    if test:
        # remove NaNs
        data = data.dropna(subset=[x, y])
        # remove categories with only one element
        keep = data.groupby(x).size()[data.groupby(x).size() > 1].index
        data = data.loc[data[x].isin(keep), :]
        if data[x].dtype.name == "category":
            data[x] = data[x].cat.remove_unused_categories()
        ylim = _ax.get_ylim()
        ylength = abs(ylim[1]) + abs(ylim[0])
        stat = pd.DataFrame(
            itertools.combinations(data[x].unique(), 2), columns=["A", "B"]
        )
        try:
            stat = pg.pairwise_ttests(data=data, dv=y, between=x, **test_kws)
        except (AssertionError, ValueError) as e:
            print(str(e))
        except KeyError:
            print("Only one category with values!")
        if multiple_testing is not False:
            if "p-unc" not in stat.columns:
                stat["p-unc"] = np.nan
            stat["p-cor"] = pg.multicomp(
                stat["p-unc"].values, method=multiple_testing
            )[1]
            pcol = "p-cor"
        else:
            pcol = "p-unc"

        # This ensures there is a point for each `x` class and keeps the order
        # correct for below
        # TODO: check for hue usage
        mm = data.groupby(x).median()
        order = stat[["A", "B"]].stack().unique()
        mm = mm.loc[order]  # sort by order
        _ax.scatter(mm.index, mm, alpha=0, color="white")

        i = 0
        for idx, row in stat.iterrows():
            p = row[pcol]
            if (pd.isnull(p) or (p > test_upper_threshold)) and (
                not plot_non_significant
            ):
                continue
            symbol = (
                "**"
                if p <= test_lower_threshold
                else "n.s."
                if ((p > test_upper_threshold) or pd.isnull(p))
                else "*"
            )
            # py = data[y].quantile(0.95) - (i * (ylength / 20))
            py = data[y].max() - (i * (ylength / 50))
            _ax.plot(
                (row["A"], row["B"]), (py, py), color="black", linewidth=1.2
            )
            _ax.text(row["B"], py, s=symbol, color="black")
            i += 1
        _ax.set_ylim(ylim)
        return (fig, stat) if ax is None else stat
    return fig if ax is None else None
