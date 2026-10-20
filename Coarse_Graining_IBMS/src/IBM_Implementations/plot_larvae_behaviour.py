"""Figure: larvae behaviour (spatial density + reserve dynamics).

A two-panel figure describing what individual larvae do inside one cycle of the
IBM:

* **Left** — the reserve ``R`` of every larva as a function of time, each line
  coloured by the larva's eventual fate (pupated / timed-out / starved), with
  the pupation threshold ``R_pup`` marked.
* **Right** — a snapshot of the square grid the larvae move on, with larval
  population density shown as a heatmap.

Run via the thin wrapper ``src/conference_figures.py --figure 1``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D

import plot_style
from IBM_Implementations.agent_based_larval_model import run_simulation

#: Colour for the post-commitment ("committed to pupation") trajectory segment
#: in the ``commit_style="uniform"`` variant (Okabe-Ito teal-green, distinct from
#: the three fate colours).
COMMIT_COLOR = "#009E73"


def _lighten(color: str, amount: float = 0.55) -> tuple[float, float, float]:
    """Blend *color* towards white by *amount* (0 = unchanged, 1 = white)."""
    r, g, b = to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def _classify_fates(state: dict, cycle: int = 0) -> dict[int, str]:
    """Map each larva_id to its fate for *cycle*.

    ``pupated``/``starved`` come from set membership in the cycle's pupated and
    deceased frames; any other larva that appears in the trajectory log simply
    reached the end of the cycle without doing either -> ``timed_out``.
    """
    pupated = state["pupated_all"][cycle]
    deceased = state["deceased_all"][cycle]
    pupated_ids = set(pupated["larva_id"]) if pupated is not None else set()
    starved_ids = set(deceased["larva_id"]) if deceased is not None else set()

    traj = state["larva_trajectories"]
    all_ids = set(traj.loc[traj["cycle"] == cycle, "larva_id"].unique())

    fates: dict[int, str] = {}
    for lid in all_ids:
        if lid in pupated_ids:
            fates[lid] = "pupated"
        elif lid in starved_ids:
            fates[lid] = "starved"
        else:
            fates[lid] = "timed_out"
    return fates


def plot_larvae_behaviour(
    seed: int = 7,
    map_size: int = 20,
    fixed_temperature: int = 20,
    feeding_type: str = "interference",
    eggs_count: int = 850,
    snapshot_hour: int | None = None,
    commit_style: str = "uniform",
    name: str = "larvae_behaviour",
    save: bool = True,
):
    """Build and (optionally) save the two-panel larvae-behaviour figure.

    A single cycle is simulated with per-larva logging enabled. *snapshot_hour*
    selects the hour whose spatial distribution is shown in the right panel; if
    ``None`` the hour with the most living larvae is used (the fullest grid).

    The default *eggs_count* seeds a crowded pond sitting at the density-
    dependence tipping point, so the cohort splits across all three fates
    (pupation, timing-out and starvation) rather than every larva pupating as it
    would under an uncrowded default cohort.

    Because pupation is a *committed* transition (a larva that reaches ``Rpup``
    keeps feeding for ``pup_delay`` hours before it actually pupates), each
    trajectory is recoloured at the hour it first reaches ``Rpup``.  *commit_style*
    controls how the post-commitment segment is drawn:

    * ``"uniform"`` — a single dedicated "committed to pupation" colour;
    * ``"faded"``   — a lightened shade of the larva's fate colour;
    * ``"none"``    — no recolouring (the whole line uses the fate colour).
    """
    state = run_simulation(
        seed=seed,
        cycles=1,
        map_size=map_size,
        fixed_temperature=fixed_temperature,
        feeding_type=feeding_type,
        random_mating=False,
        eggs_count=eggs_count,
        record_larvae=True,
    )

    traj = state["larva_trajectories"]
    traj = traj[traj["cycle"] == 0]
    Rpup = state["constants"]["Rpup"]
    fates = _classify_fates(state, cycle=0)

    # Choose the snapshot hour (fullest grid) if not specified.
    if snapshot_hour is None:
        counts_per_hour = traj.groupby("hour")["larva_id"].nunique()
        snapshot_hour = int(counts_per_hour.idxmax())

    # Build per-larva (hour, R) segments, split at the pupation-commitment point
    # (the first hour a larva reaches Rpup).  The pre-commitment ("feeding")
    # part is grouped by fate; the post-commitment ("committed") part is tracked
    # separately so it can be recoloured.  The crossing point is shared by both
    # segments so the line stays continuous.
    pre_by_fate: dict[str, list] = {k: [] for k in plot_style.FATE_COLORS}
    committed_by_fate: dict[str, list] = {k: [] for k in plot_style.FATE_COLORS}
    for lid, larva in traj.groupby("larva_id"):
        larva = larva.sort_values("hour")
        pts = np.column_stack([larva["hour"].to_numpy(), larva["R"].to_numpy()])
        fate = fates[lid]
        above = np.nonzero(pts[:, 1] >= Rpup)[0]
        if above.size == 0 or commit_style == "none":
            pre_by_fate[fate].append(pts)
        else:
            k = int(above[0])
            if k > 0:
                pre_by_fate[fate].append(pts[: k + 1])
            committed_by_fate[fate].append(pts[k:])

    max_r = float(traj["R"].max())

    # Larger type than the house default: this is a talk figure that needs to
    # read from the back of a room.
    conference_rc = {
        **plot_style.HOUSE_RC,
        "font.size": 17,
        "axes.titlesize": 19,
        "axes.labelsize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 15,
    }

    with plt.rc_context(conference_rc):
        fig, (ax_r, ax_grid) = plt.subplots(1, 2, figsize=(13, 5))

        # ---- Left panel: reserve trajectories coloured by fate ----------------
        # Pre-commitment (feeding) phase, coloured by eventual fate.  Draw in an
        # order that keeps the rarer, more salient fates on top.
        for fate in ("timed_out", "starved", "pupated"):
            segs = pre_by_fate[fate]
            if not segs:
                continue
            ax_r.add_collection(
                LineCollection(
                    segs,
                    colors=plot_style.FATE_COLORS[fate],
                    linewidths=0.8,
                    alpha=0.35,
                )
            )

        # Post-commitment ("committed to pupation") phase, drawn on top.
        if commit_style == "uniform":
            committed = [s for fate in committed_by_fate for s in committed_by_fate[fate]]
            if committed:
                ax_r.add_collection(
                    LineCollection(
                        committed, colors=COMMIT_COLOR, linewidths=0.9, alpha=0.5
                    )
                )
        elif commit_style == "faded":
            for fate in ("timed_out", "starved", "pupated"):
                segs = committed_by_fate[fate]
                if not segs:
                    continue
                ax_r.add_collection(
                    LineCollection(
                        segs,
                        colors=[_lighten(plot_style.FATE_COLORS[fate])],
                        linewidths=0.9,
                        alpha=0.7,
                    )
                )

        ax_r.axhline(
            Rpup,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label=r"$R_{\mathrm{pup}}$",
        )
        ax_r.set_xlim(0, state["constants"]["time_steps"])
        ax_r.set_ylim(0, max(Rpup * 1.08, max_r * 1.03))
        ax_r.set_xlabel(r"Time (hours)")
        ax_r.set_ylabel(r"Reserves $R$ ($\mu$g)")
        ax_r.set_title(r"Larval reserve dynamics")
        # Open the axes: drop the top and right spines (keep left/bottom).
        ax_r.spines["top"].set_visible(False)
        ax_r.spines["right"].set_visible(False)

        fate_counts = {
            fate: sum(1 for f in fates.values() if f == fate)
            for fate in plot_style.FATE_COLORS
        }
        legend_handles = [
            Line2D(
                [0],
                [0],
                color=plot_style.FATE_COLORS[fate],
                linewidth=2,
                label=f"{plot_style.FATE_LABELS[fate]} ({fate_counts[fate]})",
            )
            for fate in ("pupated", "timed_out", "starved")
        ]
        if commit_style == "uniform":
            legend_handles.append(
                Line2D(
                    [0], [0], color=COMMIT_COLOR, linewidth=2,
                    label=r"Committed to pupation",
                )
            )
        elif commit_style == "faded":
            legend_handles.append(
                Line2D(
                    [0], [0], color=_lighten(plot_style.FATE_COLORS["pupated"]),
                    linewidth=2, label=r"Committed (faded)",
                )
            )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="--",
                linewidth=1.2,
                label=r"$R_{\mathrm{pup}}$",
            )
        )
        ax_r.legend(handles=legend_handles, loc="lower right", framealpha=0.9)

        # ---- Right panel: spatial density heatmap at the snapshot hour --------
        snap = traj[traj["hour"] == snapshot_hour]
        density, _, _ = np.histogram2d(
            snap["x"].to_numpy(),
            snap["y"].to_numpy(),
            bins=map_size,
            range=[[0.5, map_size + 0.5], [0.5, map_size + 0.5]],
        )
        # Each grid cell is 1 cm wide, so the map_size x map_size grid spans
        # map_size cm in each direction; label the axes in physical cm.
        im = ax_grid.imshow(
            density.T,
            origin="lower",
            aspect="equal",
            cmap=plot_style.SEQUENTIAL_CMAP,
            extent=(0, map_size, 0, map_size),
        )
        ax_grid.set_xlabel(r"$x$ (cm)")
        ax_grid.set_ylabel(r"$y$ (cm)")
        ax_grid.set_title(
            rf"Larval density (hour {snapshot_hour}, "
            rf"$n={int(snap['larva_id'].nunique())}$)"
        )
        cbar = fig.colorbar(im, ax=ax_grid, fraction=0.046, pad=0.04)
        cbar.set_label(r"Larvae per cm$^2$")

        fig.tight_layout()
        if save:
            plot_style.save_fig(fig, name)

    return fig, state
