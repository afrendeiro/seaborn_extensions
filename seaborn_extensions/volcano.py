import numpy as np
import matplotlib.pyplot as plt

from seaborn_extensions.types import Figure, DataFrame
from seaborn_extensions.utils import get_grid_dims


def volcano_plot(
    stats: DataFrame,
    annotate_text: bool = True,
    diff_threshold: float = 0.05,
    n_top: int = None,
    invert_direction: bool = True,
) -> Figure:
    """
    Assumes stats dataframe from seaborn_extensions.swarmboxenplot:
        - "hedges" column has effect size
        - "p-unc" column has significance
        - "p-cor" column has significance (multiple test corrected)
        - A / B -> positive hedges value if A > B.
    """
    if diff_threshold is not None:
        assert n_top is None
    else:
        assert n_top is not None

    combs = stats[["A", "B"]].drop_duplicates().reset_index(drop=True)
    if invert_direction:
        stats["hedges"] *= -1  # convert to B / A which is often more intuitive
    stats["logp-unc"] = -np.log10(stats["p-unc"].fillna(1))
    stats["logp-cor"] = -np.log10(stats["p-cor"].fillna(1))
    stats["p-cor-plot"] = (stats["logp-cor"] / stats["logp-cor"].max()) * 5
    n, m = get_grid_dims(combs.shape[0])
    fig, axes = plt.subplots(n, m, figsize=(4 * m, 4 * n), squeeze=False)
    idx = -1
    for idx, (a, b) in combs.iterrows():
        ax = axes.flatten()[idx]
        p = stats.query(f"A == '{a}' & B == '{b}'")
        ax.axvline(0, linestyle="--", color="grey")
        ax.scatter(
            p["hedges"],
            p["logp-unc"],
            c=p["hedges"],
            s=5 + (2 ** p["p-cor-plot"]),
            cmap="coolwarm",
        )
        ax.set(title=f"{b} / {a}", ylabel=None, xlabel=None)
        if annotate_text:
            if diff_threshold is not None:
                ts = p.query("`p-cor` < 0.05").index
            else:
                ts = p.sort_values("p-unc").head(n_top).index
            for t in ts:
                ax.text(
                    p.loc[t, "hedges"],
                    p.loc[t, "logp-unc"],
                    s=p.loc[t, "Variable"],
                    ha="right" if p.loc[t, "hedges"] > 0 else "left",
                )
    for ax in axes.flatten()[idx + 1 :]:
        ax.axis("off")

    for ax in axes[:, 0]:
        ax.set(ylabel="-log10(p-val)")
    for ax in axes[-1, :]:
        ax.set(xlabel="Hedges' g")

    if "p-cor-plot" in stats.columns:
        stats = stats.drop("p-cor-plot", axis=1)
    return fig
