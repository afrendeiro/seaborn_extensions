#!/usr/bin/env python

"""Tests for `seaborn_extensions` package."""

import pytest

import numpy as np
import pandas as pd
import matplotlib
from seaborn_extensions import swarmboxenplot

np.random.seed(21)


@pytest.fixture
def data_simple_nodiff():
    data = pd.DataFrame(
        {"cont": np.random.random(20), "cat": np.random.choice(["a", "b"], 20)}
    ).convert_dtypes()
    return data


@pytest.fixture
def data_simple_diff():
    data = pd.DataFrame(
        [np.random.random(20), np.random.choice(["a", "b"], 20)],
        index=["cont", "cat"],
    ).T.convert_dtypes()
    data.loc[data["cat"] == "b", "cont"] *= 5
    return data


@pytest.fixture
def data_simple_long_nodiff():
    data = pd.DataFrame(
        [np.random.random(40), np.random.choice(["a", "b", "c"], 40)],
        index=["cont", "cat"],
    ).T.convert_dtypes()
    return data


@pytest.fixture
def data_simple_long_diff():
    data = pd.DataFrame(
        [np.random.random(40), np.random.choice(["a", "b", "c"], 40)],
        index=["cont", "cat"],
    ).T.convert_dtypes()
    data.loc[data["cat"] == "b", "cont"] *= 5
    data.loc[data["cat"] == "c", "cont"] -= 5
    return data


@pytest.fixture
def data_interaction():

    data = pd.DataFrame(
        {
            "x": pd.Categorical(
                np.random.choice(["a", "b", "c"], 100),
                categories=["a", "b", "c"],
                ordered=True,
            ),
            "h": pd.Categorical(
                np.random.choice(["p", "q", "r", "w"], 100),
                categories=["p", "q", "r", "w"],
                ordered=True,
            ),
            "y": np.random.random(size=100),
        }
    )
    data.loc[data["x"] == "b", "y"] += 1
    data.loc[data["x"] == "c", "y"] -= 1.5
    data.loc[(data["x"] == "c") & (data["h"] == "p"), "y"] *= 2.5
    return data


@pytest.fixture
def data_complex_missing():
    data = pd.DataFrame(
        {
            "x": pd.Categorical(
                np.random.choice(["a", "b", "c"], 100),
                categories=["a", "b", "c"],
                ordered=True,
            ),
            "y1": np.random.normal(size=100),
            "y2": np.random.random(size=100),
            "y3": np.random.random(size=100),
        }
    )[::-1]
    data.loc[data["x"] == "b", "y1"] += 3
    data.loc[data["x"] == "c", "y1"] -= 2
    data.loc[data["x"] == "b", "y2"] *= 2
    data.loc[data["x"] == "c", "y2"] *= -2
    data.loc[data["x"] == "c", "y3"] -= 5
    data.loc[data["x"] == "b", "y3"] = np.nan
    return data


def has_significant(fig):
    t = [
        c
        for c in fig.axes[0].get_children()
        if isinstance(c, matplotlib.text.Text)
    ]
    tt = [e for e in t if e.get_text() in ["*", "**"]]
    return bool(t) and bool(tt)


class TestSwarmBoxenPlot:
    def notest(self, data_simple_nodiff):
        fig = swarmboxenplot(
            data=data_simple_nodiff,
            x="cat",
            y="cont",
            test=False,
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_simple_nodiff(self, data_simple_nodiff):
        fig, stats = swarmboxenplot(
            data=data_simple_nodiff,
            x="cat",
            y="cont",
            test_kws=dict(parametric=True),
        )
        assert not has_significant(fig)

    def test_simple_diff(self, data_simple_diff):
        fig, stats = swarmboxenplot(
            data=data_simple_diff,
            x="cat",
            y="cont",
            test_kws=dict(parametric=True),
        )
        assert has_significant(fig)

    def test_complex_nodiff(self, data_simple_long_nodiff):
        fig, stats = swarmboxenplot(
            data=data_simple_long_nodiff,
            x="cat",
            y="cont",
            test_kws=dict(parametric=True),
        )
        assert not has_significant(fig)

    def test_complex_diff(self, data_simple_long_diff):
        data = data_simple_long_diff
        fig, stats = swarmboxenplot(
            data=data,
            x="cat",
            y="cont",
            test_kws=dict(parametric=True),
        )
        assert has_significant(fig)

        assert has_significant(fig)

    def test_interaction(self, data_interaction):
        data = data_interaction

        fig, stats = swarmboxenplot(
            data=data, x="x", y="y", hue="h", test_kws=dict(parametric=False)
        )

        assert has_significant(fig)
        fig, stats = swarmboxenplot(
            data=data, x="h", y="y", hue="x", test_kws=dict(parametric=False)
        )

        assert has_significant(fig)

    def test_complex_missing(self, data_complex_missing):
        data = data_complex_missing
        fig, stats = swarmboxenplot(
            data=data,
            x="x",
            y=["y1", "y2", "y3"],
            test_kws=dict(parametric=False),
        )
        assert has_significant(fig)

        fig = swarmboxenplot(
            data=data, x="x", y=["y1", "y2", "y3", "y2", "y3"], test=False
        )
        assert not has_significant(fig)

        fig = swarmboxenplot(data=data, x="x", y=["y1"], test=False)
        assert not has_significant(fig)

        with pytest.raises(ValueError):
            fig, stats = swarmboxenplot(
                data=data, x="x", y=["y1", "y2"], hue="y3"
            )
        with pytest.raises(ValueError):
            fig, stats = swarmboxenplot(data=data, x="y1", y="y2")

    def test_various_y_types(self, data_complex_missing):
        data = data_complex_missing
        fig, stats = swarmboxenplot(
            data=data,
            x="x",
            y=pd.Series(["y1", "y2", "y3"]),
            test_kws=dict(parametric=False),
        )
        assert has_significant(fig)
        fig, stats = swarmboxenplot(
            data=data,
            x="x",
            y=pd.Index(["y1", "y2", "y3"]),
            test_kws=dict(parametric=False),
        )
        assert has_significant(fig)

        fig, stats = swarmboxenplot(
            data=data,
            x="x",
            y=["y1", "y2", "y3"],
            test_kws=dict(parametric=False),
        )
        assert has_significant(fig)
