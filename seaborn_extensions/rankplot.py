import typing as tp

import matplotlib.pyplot as plt

# from plottify import autosize

from seaborn_extensions.types import Figure, Axis, Series


def rankplot(
    series: Series,
    annotate_text: tp.Union[bool, tp.Sequence[str]] = True,
    n_top: int = 10,
    diff_threshold: tp.Optional[float] = None,
    fig_kws: tp.Dict = None,
    scatter_kws: tp.Dict = None,
    ax_kws: tp.Dict = None,
    ax: Axis = None,
) -> Figure:
    """ """
    if fig_kws is None:
        fig_kws = dict(figsize=(6, 4))
        fig_kws = dict()
    if scatter_kws is None:
        scatter_kws = dict()
    if ax_kws is None:
        ax_kws = dict(xlabel="Rank", ylabel="Value")
    if ax is None:
        fig, ax = plt.subplots(**fig_kws)
    else:
        fig = ax.figure
    rank = series.rank(method="dense")
    v = series.abs().max()
    ax.scatter(rank, series, c=series, cmap="coolwarm", vmin=-v, vmax=v, **scatter_kws)

    if annotate_text is not False:
        if annotate_text is True:
            if diff_threshold is not None:
                ts = series[series.abs() > diff_threshold].index.tolist()
            else:
                ts = series.sort_values().head(n_top // 2).index.tolist()
                ts += series.sort_values().tail(n_top // 2).index.tolist()
        else:
            ts = series.loc[annotate_text].index.tolist()
        for t in ts:
            ax.text(
                rank.loc[t],
                series.loc[t],
                s=t,
                ha="left" if series.loc[t] > 0 else "right",
            )
    ax.set(**ax_kws)
    # autosize(fig)
    ax.axhline(0, color="grey", linestyle="--", linewidth=1)
    return fig
