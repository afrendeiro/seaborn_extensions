#!/usr/bin/env python

"""Tests for `seaborn_extensions` package."""

import pytest

import numpy as np
import pandas as pd
import matplotlib
from seaborn_extensions import swarmboxenplot


@pytest.fixture
def data_simple_nodiff():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    data = pd.DataFrame(
        [np.random.random(20), np.random.choice(["a", "b"], 20)],
        index=["cont", "cat"],
    ).T.convert_dtypes()
    return data


@pytest.fixture
def data_simple_diff():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    data = pd.DataFrame(
        [np.random.random(20), np.random.choice(["a", "b"], 20)],
        index=["cont", "cat"],
    ).T.convert_dtypes()
    data.loc[data["cat"] == "b", "cont"] *= 5
    return data


@pytest.fixture
def data_complex_nodiff():
    data = pd.DataFrame(
        [np.random.random(40), np.random.choice(["a", "b", "c"], 40)],
        index=["cont", "cat"],
    ).T.convert_dtypes()
    return data


@pytest.fixture
def data_complex_diff():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    data = pd.DataFrame(
        [np.random.random(40), np.random.choice(["a", "b", "c"], 40)],
        index=["cont", "cat"],
    ).T.convert_dtypes()
    data.loc[data["cat"] == "b", "cont"] *= 5
    data.loc[data["cat"] == "c", "cont"] -= 5
    return data


def is_significant(fig):
    t = [
        c
        for c in fig.axes[0].get_children()
        if isinstance(c, matplotlib.text.Text)
    ]
    tt = [e for e in t if e.get_text() == "**"]
    return bool(t) and bool(tt)


class TestSwarmBoxenPlot:
    def notest(self, data_simple_nodiff):
        """Sample pytest test function with the pytest fixture as an argument."""
        fig = swarmboxenplot(
            data=data_simple_nodiff, x="cat", y="cont", test=False,
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_simple_nodiff(self, data_simple_nodiff):
        """Sample pytest test function with the pytest fixture as an argument."""
        fig, stats = swarmboxenplot(
            data=data_simple_nodiff,
            x="cat",
            y="cont",
            test_kws=dict(parametric=True),
        )
        assert not is_significant(fig)

    def test_simple_diff(self, data_simple_diff):
        """Sample pytest test function with the pytest fixture as an argument."""
        fig, stats = swarmboxenplot(
            data=data_simple_diff,
            x="cat",
            y="cont",
            test_kws=dict(parametric=True),
        )
        assert is_significant(fig)

    def test_complex_nodiff(self, data_complex_nodiff):
        """Sample pytest test function with the pytest fixture as an argument."""
        fig, stats = swarmboxenplot(
            data=data_complex_nodiff,
            x="cat",
            y="cont",
            test_kws=dict(parametric=True),
        )
        assert not is_significant(fig)

    def test_complex_diff(self, data_complex_diff):
        """Sample pytest test function with the pytest fixture as an argument."""
        fig, stats = swarmboxenplot(
            data=data_complex_diff,
            x="cat",
            y="cont",
            test_kws=dict(parametric=True),
        )
        assert is_significant(fig)
