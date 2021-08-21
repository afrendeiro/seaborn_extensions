"""
A type of plot that combines swarms and box(en)/bar plots in an overlaid fashion.
"""

import typing as tp
import itertools
import warnings

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg

from seaborn_extensions.types import DataFrame, Axis, Figure, Iterables
from seaborn_extensions.utils import get_grid_dims, filter_kwargs_by_callable


def swarmboxenplot(
    data: DataFrame,
    x: str,
    y: tp.Union[str, Iterables],
    hue: str = None,
    swarm: bool = True,
    boxen: bool = True,
    bar: bool = False,
    ax: tp.Union[Axis, tp.Sequence[Axis]] = None,
    test: bool = True,
    multiple_testing: tp.Union[bool, str] = "fdr_bh",
    test_upper_threshold: float = 0.05,
    test_lower_threshold: float = 0.01,
    plot_non_significant: bool = False,
    plot_kws: tp.Dict[str, tp.Any] = None,
    test_kws: tp.Dict[str, tp.Any] = None,
    fig_kws: tp.Dict[str, tp.Any] = None,
) -> tp.Optional[tp.Union[Figure, DataFrame, tp.Tuple[Figure, DataFrame]]]:
    """
    A categorical plot that overlays individual observations
    as a swarm plot and summary statistics about them in a boxen plot.

    In addition, this plot will test differences between observation
    groups and add lines representing a significant difference between
    them.

    Parameters
    ----------
    data: pd.DataFrame
        A dataframe with data where the rows are the observations and
        columns are the variables to group them by.
    x: str
        The categorical variable.
    y: str | list[str]
        The continuous variable to plot.
        If more than one is given, will ignore the `ax` attribute and
        return figure with a subplot per each `y` variable.
    hue: str, optional
        An optional categorical variable to further group observations by.
    swarm: bool
        Whether to plot individual observations as a swarmplot.
    boxen: bool
        Whether to plot summary statistics as a boxenplot.
    ax: matplotlib.axes.Axes, optional
        An optional axes to draw in.
    test: bool
        Whether to test differences between observation groups.
        If `False`, will not return a dataframe as well.
    multiple_testing: str
        Method for multiple testing correction.
    test_upper_threshold: float
        Upper theshold to consider p-values significant.
        Will be marked with "*".
    test_lower_threshold: float
        Secondary theshold to consider p-values highly significant.
        Will be marked with "**".
    plot_non_significant: bool
        Whether to add a "n.s." sign to p-values above `test_upper_threshold`.
    plot_kws: dict
        Additional values to pass to seaborn.boxenplot or seaborn.swarmplot
    test_kws: dict
        Additional values to pass to pingouin.pairwise_ttests.
        The default is: dict(parametric=False) to run a non-parametric test.

    Returns
    -------
    tuple[Figure, pandas.DataFrame]:
        if `ax` is None and `test` is True.
    pandas.DataFrame:
        if `ax` is not None.
    Figure:
        if `test` is False.
    None:
        if `test` is False and `ax` is not None.

    Raises
    ------
    ValueError:
        If either the `x` or `hue` column in `data` are not
        Category, string or object type, or if `y` is not numeric.

    """
    # opts = dict(data=data, x='h', y='y', hue='x', test_kws=dict(parametric=False))
    # opts = dict(data=data, x='cat', y='cont')
    # for k, v in opts.items():
    #     locals()[k] = v

    for var, name in [(x, "x"), (hue, "hue")]:
        if var is not None:
            if not data[var].dtype.name in ["category", "string", "object"]:
                raise ValueError(
                    f"`{name}` variable must be categorical, string or object."
                )

    if test_kws is None:
        test_kws = dict(parametric=False)
    if plot_kws is None:
        plot_kws = dict()

    data = data.sort_values([x] + ([hue] if hue is not None else []))

    if not isinstance(y, str):
        # TODO: display only one legend for hue
        if ax is None:
            n, m = get_grid_dims(y)
            default_fig_kws = dict(
                nrows=n, ncols=m, figsize=(m * 4, n * 4), sharex=True, squeeze=False
            )
            default_fig_kws.update(fig_kws or {})
            fig, axes = plt.subplots(**default_fig_kws)
            axes = axes.flatten()
        else:
            if isinstance(ax, np.ndarray):
                axes = ax.flatten()

        _stats = list()
        idx = -1
        for idx, _var in enumerate(y):
            _ax = axes[idx]
            s: DataFrame = swarmboxenplot(
                data=data,
                x=x,
                y=_var,
                hue=hue,
                swarm=swarm,
                boxen=boxen,
                bar=bar,
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
            if test:
                _stats.append(s.assign(Variable=_var))
        # "close" excess subplots
        for _ax in axes[idx + 1 :]:
            _ax.axis("off")
        if test:
            stats = pd.concat(_stats).reset_index(drop=True)
            cols = [c for c in stats.columns if c != "Variable"]
            stats = stats.reindex(["Variable"] + cols, axis=1)

            # If there is just one test per `y` (no hue), correct p-values
            if stats.shape == len(y):
                stats["p-cor"] = pg.multicomp(
                    stats["p-unc"].values, method=multiple_testing
                )[1]
        if ax is None:
            return (fig, stats) if test else fig
        return stats if test else None

    if data[y].dtype.name in ["category", "string", "object"]:
        raise ValueError("`y` variable must be numeric.")

    if ax is None:
        fig, _ax = plt.subplots(1, 1, figsize=(4, 4))
    else:
        _ax = ax

    # Plot vanilla seaborn
    if boxen:
        assert not bar
        # Tmp fix for lack of support for Pandas Int64 in boxenplot:
        if data[y].dtype.name == "Int64":
            data[y] = data[y].astype(float)
        boxen_kws = filter_kwargs_by_callable(plot_kws, sns.boxenplot)
        sns.boxenplot(data=data, x=x, y=y, hue=hue, ax=_ax, **boxen_kws)
    if bar:
        assert not boxen
        bar_kws = filter_kwargs_by_callable(plot_kws, sns.barplot)
        sns.barplot(data=data, x=x, y=y, hue=hue, ax=_ax, **bar_kws)

    if (boxen or bar) and swarm:
        _add_transparency_to_plot(_ax, kind="bar" if bar else "boxen")
    if swarm:
        swarm_kws = filter_kwargs_by_callable(plot_kws, sns.swarmplot)
        if hue is not None and "dodge" not in swarm_kws:
            swarm_kws["dodge"] = True
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            sns.swarmplot(data=data, x=x, y=y, hue=hue, ax=_ax, **swarm_kws)
    _ax.set_xticklabels(_ax.get_xticklabels(), rotation=90, ha="right")

    if test:
        #
        if not data.index.is_unique:
            print("Warning: dataframe contains a duplicated index.")

        # remove NaNs
        datat = data.dropna(subset=[x, y] + ([hue] if hue is not None else []))
        # remove categories with only one element
        keep = datat.groupby(x).size()[datat.groupby(x).size() > 1].index
        datat = datat.loc[datat[x].isin(keep), :]
        if datat[x].dtype.name == "category":
            datat[x] = datat[x].cat.remove_unused_categories()
        ylim = _ax.get_ylim()  # save original axis boundaries for later
        ylength = abs(ylim[1]) + (abs(ylim[0]) if ylim[0] < 0 else 0)

        # Now calculate stats
        # # get empty dataframe in case nothing can be calculated
        stat = _get_empty_stat_results(datat, x, y, hue, add_median=True)
        # # mirror groups to account for own pingouin order
        stat = (
            stat.append(
                stat.rename(
                    columns={
                        "B": "A",
                        "A": "B",
                        "median_A": "median_B",
                        "median_B": "median_A",
                    }
                )
            )
            .sort_values(["Contrast", "A", "B"])
            .reset_index(drop=True)
        )
        try:
            _stat = pg.pairwise_ttests(
                data=datat,
                dv=y,
                between=x if hue is None else [x, hue],
                **test_kws,
            )
            stat = _stat.merge(
                stat[
                    ["Contrast", "A", "B", "median_A", "median_B"]
                    + ([x] if hue is not None else [])
                ],
                how="left",
            )
        except (AssertionError, ValueError) as e:
            print(str(e))
        except KeyError:
            print("Only one category with values!")
        if multiple_testing is not False:
            if "p-unc" not in stat.columns:
                stat["p-unc"] = np.nan
            stat["p-cor"] = pg.multicomp(stat["p-unc"].values, method=multiple_testing)[1]
            pcol = "p-cor"
        else:
            pcol = "p-unc"

        # This ensures there is a point for each `x` class and keep the order
        # correct for below
        mm = data.groupby([x] + ([hue] if hue is not None else []))[y].median()
        if hue is None:
            order = {k: float(i) for i, k in enumerate(mm.index)}
        else:
            nhues = data[hue].drop_duplicates().dropna().shape[0]
            order = {
                k: (float(i) / nhues) - (1 / nhues) - 0.05 for i, k in enumerate(mm.index)
            }
        _ax.scatter(order.values(), mm, alpha=0, color="white")

        # Plot significance bars
        # start at top of the plot and progressively decrease sig. bar downwards
        py = data[y].max()
        incr = ylength / 100  # divide yaxis in 100 steps
        for idx, row in stat.iterrows():
            p = row[pcol]
            if (pd.isnull(p) or (p > test_upper_threshold)) and (
                not plot_non_significant
            ):
                py -= incr
                continue
            symbol = (
                "**"
                if p <= test_lower_threshold
                else "n.s."
                if ((p > test_upper_threshold) or pd.isnull(p))
                else "*"
            )
            if hue is not None:
                if row[x] != "-":
                    xx = (order[(row[x], row["A"])], order[(row[x], row["B"])])
                else:
                    try:
                        # TODO: get more accurate middle of group
                        xx = (
                            order[(row["A"], stat["A"].iloc[-1])] - (1 / nhues),
                            order[(row["B"], stat["B"].iloc[-1])] - (1 / nhues),
                        )
                    except KeyError:
                        # These are the hue groups without contrasting on 'x'
                        continue
            else:
                xx = (order[row["A"]], order[row["B"]])

            red_fact = 0.95  # make the end position shorter
            _ax.plot(
                (xx[0], xx[1] * red_fact),
                (py, py),
                color="black",
                linewidth=1.2,
            )
            _ax.text(xx[1] * red_fact, py, s=symbol, color="black", ha="center")
            py -= incr
        _ax.set_ylim(ylim)
        return (fig, stat) if ax is None else stat
    return fig if ax is None else None


def _add_transparency_to_plot(ax: Axis, alpha: float = 0.25, kind: str = "boxen") -> None:

    objs = (
        (
            matplotlib.collections.PatchCollection,
            matplotlib.collections.PathCollection,
        )
        if kind == "boxen"
        else (matplotlib.patches.Rectangle)
    )

    for x in ax.get_children():
        if isinstance(x, objs):
            x.set_alpha(alpha)


def _get_empty_stat_results(
    data: DataFrame,
    x: str,
    y: str,
    hue: tp.Optional[str] = None,
    add_median: bool = True,
) -> DataFrame:
    stat = pd.DataFrame(
        itertools.combinations(data[x].drop_duplicates(), 2),
        columns=["A", "B"],
    )
    stat["Contrast"] = x
    if hue is not None:
        huestat = pd.DataFrame(
            itertools.combinations(data[hue].drop_duplicates(), 2),
            columns=["A", "B"],
        )
        huestat["Contrast"] = hue
        to_append = [huestat]
        for v in data[x].unique():
            n = huestat.copy()
            n[x] = v
            n["Contrast"] = f"{x} * {hue}"
            to_append.append(n)
        stat = (
            stat.append(to_append, ignore_index=True)
            .fillna("-")
            .sort_values([x, "A", "B"])
        )
    stat["Tested"] = False
    stat["p-unc"] = np.nan

    if add_median:
        mm = data.groupby(x)[y].median().reset_index()
        if hue is not None:
            mm = mm.rename(columns={x: hue})
            mm = mm.append(data.groupby(hue)[y].median().reset_index())
            mm = mm.append(data.groupby([x, hue])[y].median().reset_index()).fillna("-")
            # mm = mm.append(data.groupby([x, hue])[y].std().reset_index()).fillna("-")
        for col in ["A", "B"]:
            stat = stat.merge(
                mm.rename(
                    columns={
                        hue if hue is not None else x: f"{col}",
                        y: f"median_{col}",
                    }
                ),
                how="left",
            )
    return stat
