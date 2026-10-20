"""In-model behaviour helpers for the agent-based larval model.

Functions governing per-timestep dynamics: temperature perturbation,
spatial movement, the two feeding competition modes (interference
and exploitation), and the RDG inner ODE with its numerical integrator.
"""

import numpy as np
from SyncRNG import SyncRNG
from IBM_Implementations.sync_rng_random_functions import sync_rng_choice


def update_temperature(temp):
    """Small random perturbation to temperature (uniform [-0.1, 0.1])."""
    return temp + np.random.uniform(-0.1, 0.1)


def movement(inds_df, map_size, random_generator: SyncRNG | None = None):
    """
    Random movement of individuals. Keeps coordinates in 1..map_size (wrap-around).
    inds_df must have integer columns 'x' and 'y'. Returns a copy.
    """
    if inds_df is None or inds_df.shape[0] == 0:
        return inds_df
    inds = inds_df.copy()
    n = inds.shape[0]
    move_d = np.array([-1, 0, 1])
    if random_generator is not None:
        moves_x = sync_rng_choice(move_d, s=random_generator, size=n, replace=True)
        moves_y = sync_rng_choice(move_d, s=random_generator, size=n, replace=True)
    else:
        moves_x = np.random.choice(move_d, size=n, replace=True)
        moves_y = np.random.choice(move_d, size=n, replace=True)
    inds["x"] = ((inds["x"] + moves_x - 1) % map_size) + 1
    inds["y"] = ((inds["y"] + moves_y - 1) % map_size) + 1
    return inds


def apply_feeding_interference(
    larvae_df,
    grid_food,
    temperature,
    param_funs,
    Rth,
    sigma,
    assimilation_efficacy,
    max_competitors=10,
    random_generator: SyncRNG | None = None,
):
    """
    Determine FA_actual for each larva in larvae_df given grid_food and temperature.
    - larvae_df: DataFrame with columns 'x','y','W','R' (1-based coordinates).
    - grid_food: 2D numpy array shape (map_size, map_size) (python 0-based).
    - param_funs: dict with interpolator functions; uses 'g_fun' to compute FA threshold.
    - Rth, sigma: parameters for the FA threshold calculation
    - random_generator: SyncRNG instance for reproducible randomness (optional)
    - Returns: (larvae_df_with_FA_actual, updated_grid_food)
    Side-effects: none (works on a copy of larvae_df, returns new arrays).
    """
    if larvae_df is None or larvae_df.shape[0] == 0:
        # nothing to do
        return larvae_df, grid_food

    larvae = larvae_df.copy()
    larvae["FA_actual"] = 0.0

    larvae_locations = larvae.loc[:, "x":"y"].to_numpy()
    occupied_cells = larvae.loc[:, "x":"y"].drop_duplicates().to_numpy()
    larvae_indices = larvae.index.to_numpy()

    # For each occupied cell determine winners and allocate food
    for x, y in zip(occupied_cells[:, 0], occupied_cells[:, 1]):
        idx_in_cell = larvae_indices[
            (larvae_locations[:, 0] == x) & (larvae_locations[:, 1] == y)
        ]
        n_in = idx_in_cell.shape[0]
        if n_in == 0:
            continue

        # Choose winners (up to max_competitors) by W-weighted probability if > max_competitors
        if n_in > max_competitors:
            weights = larvae.loc[idx_in_cell, "W"].values
            # when weights sum to zero, fallback to uniform
            if weights.sum() > 0:
                probs = weights / weights.sum()
            else:
                probs = np.repeat(1.0 / weights.size, weights.size)

            if random_generator is not None:
                winners = list(
                    sync_rng_choice(
                        idx_in_cell,
                        s=random_generator,
                        size=max_competitors,
                        replace=False,
                        weights=probs,
                    )
                )
            else:
                winners = list(
                    np.random.choice(
                        idx_in_cell, size=max_competitors, replace=False, p=probs
                    )
                )
        else:
            winners = list(idx_in_cell)

        # allocate from the pool in descending W order (matching R behaviour)
        winners_sorted = sorted(
            winners, key=lambda ii: larvae.at[ii, "W"], reverse=True
        )

        pool = float(grid_food[int(x - 1), int(y - 1)])  # convert to 0-based index

        for ii in winners_sorted:
            FA_desired_vec = (
                param_funs["g_fun"](temperature)
                * larvae.at[ii, "g_fac"]
                * larvae.at[ii, "W"]
                * np.exp(-((larvae.at[ii, "R"] - Rth) ** 2) / (2 * (sigma**2)))
            )
            FA_th = FA_desired_vec / assimilation_efficacy
            if pool >= FA_th:
                larvae.at[ii, "FA_actual"] = FA_desired_vec
                pool -= FA_th
            else:
                # not enough pool left -> stop allocating in that cell
                break

        # write updated pool back into grid_food
        grid_food[int(x - 1), int(y - 1)] = pool

    return larvae, grid_food


def apply_feeding_exploitation(
    larvae_df, grid_food, temperature, param_funs, Rth, sigma, assimilation_efficacy
):
    """
    Determine FA_actual for each larva in larvae_df given grid_food and temperature.
    - larvae_df: DataFrame with columns 'x','y','W','R' (1-based coordinates).
    - grid_food: 2D numpy array shape (map_size, map_size) (python 0-based).
    - param_funs: dict with interpolator functions; uses 'g_fun' to compute FA threshold.
    - Returns: (larvae_df_with_FA_actual, updated_grid_food)
    Side-effects: none (works on a copy of larvae_df, returns new arrays).
    """
    if larvae_df is None or larvae_df.shape[0] == 0:
        # nothing to do
        return larvae_df, grid_food

    larvae = larvae_df.copy()
    larvae["FA_actual"] = 0.0

    larvae_locations = larvae.loc[:, "x":"y"].to_numpy()
    occupied_cells = larvae.loc[:, "x":"y"].drop_duplicates().to_numpy()
    larvae_indices = larvae.reset_index(drop=True).index.to_numpy()

    # precompute values that we will need repeatedly in the loop (less indexing of arrays gives much better performance)
    g_values = param_funs["g_fun"](temperature) * larvae.loc[:, "g_fac"]
    FA_desired_all = (
        g_values
        * larvae.loc[:, "W"]
        * np.exp(-((larvae.loc[:, "R"] - Rth) ** 2) / (2 * sigma**2))
    ).values
    FA_actual = np.empty_like(FA_desired_all)

    # For each occupied cell determine winners and allocate food
    for x, y in zip(occupied_cells[:, 0], occupied_cells[:, 1]):
        idx_in_cell = larvae_indices[
            (larvae_locations[:, 0] == x) & (larvae_locations[:, 1] == y)
        ]
        n_in = idx_in_cell.shape[0]
        if n_in == 0:
            continue

        FA_desired = FA_desired_all[idx_in_cell]

        per_larvae_food_requirements = FA_desired / assimilation_efficacy
        total_requirements = np.sum(per_larvae_food_requirements)

        if total_requirements <= grid_food[int(x - 1), int(y - 1)]:
            FA_actual[idx_in_cell] = FA_desired
            grid_food[int(x - 1), int(y - 1)] = max(
                0,
                grid_food[int(x - 1), int(y - 1)]
                - np.sum(per_larvae_food_requirements),
            )
        else:
            per_larvae_proportion_of_food_demand = (
                per_larvae_food_requirements / total_requirements
            )
            share_of_food_obtained = (
                grid_food[int(x - 1), int(y - 1)] * per_larvae_proportion_of_food_demand
            )

            actual_food_obtained = np.minimum(
                share_of_food_obtained * assimilation_efficacy, FA_desired
            )
            FA_actual[idx_in_cell] = actual_food_obtained
            grid_food[int(x - 1), int(y - 1)] = max(
                0,
                grid_food[int(x - 1), int(y - 1)]
                - np.sum(actual_food_obtained / assimilation_efficacy),
            )
    larvae.loc[:, "FA_actual"] = FA_actual

    # cleanup helper column
    return larvae, grid_food


def rdg_model(state: np.ndarray, pars: np.ndarray, FA=np.ndarray | None) -> np.ndarray:
    """
    RDG inner ODE. Returns dstate/dt = [dW, dR].
    state: 2xN array, where the first column is W and the second column is R
    pars: dict with keys a, c, alpha, beta, g, Rth, sigma
    pars: array of parameter values with order [a, c, alpha, beta, g, Rth, sigma]
    """
    state = state.reshape((-1, state.shape[-1]))
    W = state[:, 0]
    R = state[:, 1]
    a = pars[:, 0]
    c = pars[:, 1]
    alpha = pars[:, 2]
    beta = pars[:, 3]
    g = pars[:, 4]
    Rth = pars[:, 5]
    sigma = pars[:, 6]

    if FA is None:  # following Ziqian's logic although I suspect this is a bug
        FA = g * W * np.exp(-((R - Rth) ** 2) / (2 * sigma**2))
    M = c * (W**beta)

    out = np.empty_like(state)
    surplus = FA - M
    out[surplus < 0] = np.column_stack([np.zeros_like(surplus), surplus])[surplus < 0]
    out[surplus >= 0] = np.column_stack(
        [
            surplus[surplus >= 0]
            - a[surplus >= 0] * (surplus[surplus >= 0] ** alpha[surplus >= 0]),
            a[surplus >= 0] * (surplus[surplus >= 0] ** alpha[surplus >= 0]),
        ]
    )

    return out


def state_update(
    state_vec: np.ndarray, pars: np.ndarray, model_func, FA: np.ndarray | None = None
) -> np.ndarray:
    """
    Advance time once step, incrementing weights and reserves using the RDG inner model.
    - state_vec: np.array([W, R])
    - pars: pars dict for rdg_model
    - model_func: callable(state, pars)->dstate
    Returns: new_state (np.array([W_new, R_new]))
    """
    k1 = model_func(state_vec, pars, FA)
    k2 = model_func(state_vec + 0.5 * k1, pars, FA)
    k3 = model_func(state_vec + 0.5 * k2, pars, FA)
    k4 = model_func(state_vec + k3, pars, FA)
    delta = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    return state_vec + delta