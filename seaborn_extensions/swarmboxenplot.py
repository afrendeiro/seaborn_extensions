"""Main module."""


from typing import Any, Tuple, Union, Dict, Optional, Collection
import itertools

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg


from seaborn_extensions.types import DataFrame, Axis, Figure
from seaborn_extensions.utils import get_grid_dims


def add_transparency_to_boxenplot(ax: Axis, alpha: float = 0.25) -> None:
    patches = (
        matplotlib.collections.PatchCollection,
        matplotlib.collections.PathCollection,
    )
    for x in ax.get_children():
        if isinstance(x, patches):
            x.set_alpha(alpha)


def swarmboxenplot(
    data: DataFrame,
    x: str,
    y: Union[str, Collection] = None,
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
    fig, stats = swarmboxenplot(data=data, x='cat', y='cont')

    data = pd.DataFrame({
        "x": pd.Categorical(
            np.random.choice(['a', 'b', 'c'], 100),
            categories=['a', 'b', 'c'], ordered=True),
        "y1": np.random.normal(size=100),
        "y2": np.random.random(size=100),
        "y3": np.random.random(size=100)})[::-1]
    data.loc[data['x'] == 'b', 'y1'] += 3
    data.loc[data['x'] == 'c', 'y1'] -= 2
    data.loc[data['x'] == 'b', 'y2'] *= 2
    data.loc[data['x'] == 'c', 'y2'] *= -2
    data.loc[data['x'] == 'c', 'y3'] -= 5
    data.loc[data['x'] == 'b', 'y3'] = np.nan
    fig, stats = swarmboxenplot(data=data, x='x', y=['y1', 'y2', 'y3'], test_kws=dict(parametric=False))


    """
    if test_kws is None:
        test_kws = dict()
    if plot_kws is None:
        plot_kws = dict()

    if isinstance(y, (list, np.ndarray, pd.Series, pd.Index)):
        n, m = get_grid_dims(y)
        fig, axes = plt.subplots(n, m, figsize=(m * 4, n * 4), sharex=True)
        _stats = list()
        for i, _var in enumerate(y):
            _ax = axes.flatten()[i]
            s: DataFrame = swarmboxenplot(
                data=data,
                x=x,
                y=_var,
                hue=hue,
                swarm=swarm,
                boxen=boxen,
                ax=_ax,
                test=test,
                multiple_testing=multiple_testing,
                test_upper_threshold=test_upper_threshold,
                test_lower_threshold=test_lower_threshold,
                plot_non_significant=plot_non_significant,
                plot_kws=plot_kws,
                test_kws=test_kws,
            )
            _ax.set(title=_var, xlabel=None, ylabel=None)
            _stats.append(s.assign(Variable=_var))
        # "close" excess subplots
        for ax in axes.flatten()[i + 1 :]:
            ax.axis("off")
        stats = pd.concat(_stats).reset_index(drop=True)
        stats = stats.reindex(["Variable"] + s.columns.tolist(), axis=1)
        return fig, stats

    if ax is None:
        fig, _ax = plt.subplots(1, 1, figsize=(4, 4))
    else:
        _ax = ax
    if boxen:
        sns.boxenplot(data=data, x=x, y=y, hue=hue, ax=_ax, **plot_kws)
    if boxen and swarm:
        add_transparency_to_boxenplot(_ax)
    if swarm:
        if hue is not None and "dodge" not in plot_kws:
            plot_kws["dodge"] = True
        sns.swarmplot(data=data, x=x, y=y, hue=hue, ax=_ax, **plot_kws)
    _ax.set_xticklabels(_ax.get_xticklabels(), rotation=90)

    if test:
        # remove NaNs
        datat = data.dropna(subset=[x, y])
        # remove categories with only one element
        keep = datat.groupby(x).size()[datat.groupby(x).size() > 1].index
        datat = datat.loc[datat[x].isin(keep), :]
        if datat[x].dtype.name == "category":
            datat[x] = datat[x].cat.remove_unused_categories()
        ylim = _ax.get_ylim()
        ylength = abs(ylim[1]) + abs(ylim[0])
        stat = pd.DataFrame(
            itertools.combinations(datat[x].unique(), 2), columns=["A", "B"]
        )
        try:
            stat = pg.pairwise_ttests(
                data=datat,
                dv=y,
                between=x if hue is None else [x, hue],
                **test_kws
            )
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

        # This ensures there is a point for each `x` class and keep the order
        # correct for below
        mm = data.groupby(x)[y].median()
        _ax.scatter(mm.index, mm, alpha=0, color="white")
        order = {k: i for i, k in enumerate(mm.index)}

        # Plot significance bars
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
            py = data[y].max() - (i * (ylength / 100))
            _ax.plot(
                (order[row["A"]], order[row["B"]]),
                (py, py),
                color="black",
                linewidth=1.2,
            )
            _ax.text(order[row["B"]], py, s=symbol, color="black")
            i += 1
        _ax.set_ylim(ylim)
        return (fig, stat) if ax is None else stat
    return fig if ax is None else None
