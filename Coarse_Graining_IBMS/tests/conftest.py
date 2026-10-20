"""Test-wide matplotlib setup: headless, and never through LaTeX.

The house style renders every figure with ``text.usetex``
(:data:`plot_style.HOUSE_RC`), which needs a system TeX installation that CI
does not have.  Tests care that a figure is *built* -- the grid, the labels, the
legend -- not that its glyphs are Computer Modern, so the TeX round-trip is
switched off here.  Production figure code is untouched and still renders
through real LaTeX.
"""

import matplotlib
import matplotlib.pyplot as plt
import pytest

import plot_style

matplotlib.use("Agg", force=True)


@pytest.fixture(autouse=True)
def _render_without_tex(monkeypatch):
    """Render figures with mathtext for the duration of each test.

    ``IBM_Emulation.surrogates`` sets usetex globally at import time, so the
    live rcParams are patched as well as the house-style block.  ``setitem``
    restores both afterwards, so nothing leaks between tests.
    """
    monkeypatch.setitem(plot_style.HOUSE_RC, "text.usetex", False)
    monkeypatch.setitem(matplotlib.rcParams, "text.usetex", False)
    yield
    plt.close("all")
