"""Logging helpers for agent_based_larval_model.

Functions that collect per-hour larval statistics and append rows to the two
time-series outputs (larval_time_series and pupation_stats).  Extracted from
run_cycle to keep simulation mechanics and bookkeeping separate.
"""

import numpy as np
import pandas as pd


def collect_larval_stats(
    larvae_df: pd.DataFrame,
    current_larvae: int,
    grid_food: np.ndarray,
    map_size: int,
) -> dict:
    """Compute per-hour summary statistics from the current larvae state.

    Must be called *after* movement and *before* feeding so that spatial and
    physiological values reflect the start-of-hour population.

    Parameters
    ----------
    larvae_df:
        Current larvae DataFrame (post-movement, pre-feeding).
    current_larvae:
        Number of living larvae this hour (``larvae_df.shape[0]``).
    grid_food:
        Food grid at the current hour (before feeding depletes it).
    map_size:
        Side length of the square spatial grid.

    Returns
    -------
    dict with keys:
        occupied_cells, mean_larvae_per_cell, food_mean_cell,
        reserve_mean/median/min/max, mass_mean/median/min/max
    """
    food_mean_cell = float(grid_food.sum()) / (map_size**2)

    if current_larvae > 0:
        occupied_cells = larvae_df.loc[:, "x":"y"].drop_duplicates().shape[0]
        mean_larvae_per_cell = current_larvae / occupied_cells
        r_vals = larvae_df["R"].values
        w_vals = larvae_df["W"].values
        return {
            "occupied_cells": occupied_cells,
            "mean_larvae_per_cell": float(mean_larvae_per_cell),
            "food_mean_cell": food_mean_cell,
            "reserve_mean": float(np.mean(r_vals)),
            "reserve_median": float(np.median(r_vals)),
            "reserve_min": float(np.min(r_vals)),
            "reserve_max": float(np.max(r_vals)),
            "mass_mean": float(np.mean(w_vals)),
            "mass_median": float(np.median(w_vals)),
            "mass_min": float(np.min(w_vals)),
            "mass_max": float(np.max(w_vals)),
        }
    else:
        return {
            "occupied_cells": 0,
            "mean_larvae_per_cell": np.nan,
            "food_mean_cell": food_mean_cell,
            "reserve_mean": np.nan,
            "reserve_median": np.nan,
            "reserve_min": np.nan,
            "reserve_max": np.nan,
            "mass_mean": np.nan,
            "mass_median": np.nan,
            "mass_min": np.nan,
            "mass_max": np.nan,
        }


def append_hourly_row(
    state: dict,
    cycle: int,
    hour: int,
    current_larvae: int,
    hour_stats: dict,
    survival_rate: float,
    new_commitments: int,
    pupated_now: int,
    death_count: int,
) -> None:
    """Build one row of larval_time_series and append it to state.

    Parameters
    ----------
    state:
        Simulation state dict (mutated via ``state["larval_time_series"]``).
    cycle:
        Zero-based cycle index.
    hour:
        One-based hour index within the cycle.
    current_larvae:
        Larval count at the start of this hour, before deaths or pupation.
    hour_stats:
        Dict returned by ``collect_larval_stats``.
    survival_rate:
        Fraction of larvae alive vs. the previous hour (nan for hour 1).
    new_commitments:
        Larvae that crossed the pupation-reserve threshold this hour.
    pupated_now:
        Larvae that completed pupation this hour.
    death_count:
        Larvae that died this hour.
    """
    row = {
        "Cycle": cycle + 1,
        "Hour": hour,
        "LarvaeCount": int(current_larvae),
        "SurvivalRate": float(survival_rate) if not np.isnan(survival_rate) else np.nan,
        "PupationCommitments": int(new_commitments),
        "Pupated": int(pupated_now),
        "Deaths": int(death_count),
        "ReserveMean": hour_stats["reserve_mean"],
        "ReserveMedian": hour_stats["reserve_median"],
        "ReserveMin": hour_stats["reserve_min"],
        "ReserveMax": hour_stats["reserve_max"],
        "MassMean": hour_stats["mass_mean"],
        "MassMedian": hour_stats["mass_median"],
        "MassMin": hour_stats["mass_min"],
        "MassMax": hour_stats["mass_max"],
        "OccupiedCells": int(hour_stats["occupied_cells"]),
        "MeanLarvaePerCell": (
            float(hour_stats["mean_larvae_per_cell"])
            if not np.isnan(hour_stats["mean_larvae_per_cell"])
            else np.nan
        ),
        "MeanFoodPerCell": float(hour_stats["food_mean_cell"]),
    }
    state["larval_time_series"] += [row]


def append_cycle_stats_row(
    state: dict,
    cycle: int,
    larvae_df: pd.DataFrame,
    cycle_start: int,
    grid_food: np.ndarray,
) -> None:
    """Build one row of pupation_stats and append it to state.

    Parameters
    ----------
    state:
        Simulation state dict (reads ``pupated_all`` / ``deceased_all``,
        mutates ``pupation_stats`` in-place).
    cycle:
        Zero-based cycle index.
    larvae_df:
        Remaining (non-pupated, living) larvae at end of cycle.
    cycle_start:
        Global time-step offset for this cycle (``cycle * (time_steps + 10)``).
    grid_food:
        Food grid at the end of the cycle.
    """
    pupated = (
        state["pupated_all"][cycle]
        if state["pupated_all"][cycle] is not None
        else pd.DataFrame(columns=["W"])
    )
    non_pupated = (
        larvae_df
        if (larvae_df is not None and not larvae_df.empty)
        else pd.DataFrame(columns=["W"])
    )
    deceased = (
        state["deceased_all"][cycle]
        if state["deceased_all"][cycle] is not None
        else pd.DataFrame(columns=["W"])
    )
    pupation_times_cycle = (
        (pupated["pupate_at"] - cycle_start)
        if (not pupated.empty and "pupate_at" in pupated.columns)
        else np.array([])
    )

    stats_row = {
        "Cycle": cycle + 1,
        "PupationTimeMean": (
            float(np.mean(pupation_times_cycle))
            if pupation_times_cycle.size > 0
            else np.nan
        ),
        "PupationTimeMedian": (
            float(np.median(pupation_times_cycle))
            if pupation_times_cycle.size > 0
            else np.nan
        ),
        "PupationMassMean": float(pupated["W"].mean()) if not pupated.empty else np.nan,
        "PupationMassMedian": (
            float(pupated["W"].median()) if not pupated.empty else np.nan
        ),
        "NonPupatedMassMean": (
            float(non_pupated["W"].mean()) if not non_pupated.empty else np.nan
        ),
        "NonPupatedMassMedian": (
            float(non_pupated["W"].median()) if not non_pupated.empty else np.nan
        ),
        "DeadMassMean": float(deceased["W"].mean()) if not deceased.empty else np.nan,
        "DeadMassMedian": (
            float(deceased["W"].median()) if not deceased.empty else np.nan
        ),
        "EndFoodLevel": float(grid_food.sum()),
    }

    state["pupation_stats"] += [stats_row]
