"""Model and larvae configuration helpers for the agent-based larval model.

Functions for building temperature-dependent parameter interpolators,
survival rates, and per-larva parameter construction.
"""

import numpy as np


def build_constant_functions(fixed_temperature) -> tuple[dict, np.ndarray]:
    """
    Build constant functions for parameters (this is a useful approach as it saves me from changing the internal logic of run_cycle).
    Returns a dict of callables: a_fun, c_fun, alpha_fun, beta_fun, g_fun.
    """
    temps_all = [20, 22, 24, 26, 28, 30]

    a_vals = [0.16, 0.18, 0.15, 0.15, 0.25, 0.31]
    c_vals = [0.046, 0.033, 0.035, 0.041, 0.052, 0.069]
    alpha_vals = [0.73, 0.81, 0.96, 1.01, 0.81, 0.69]
    beta_vals = [0.36, 0.54, 0.58, 0.50, 0.50, 0.52]
    g_vals = [0.23, 0.34, 0.40, 0.44, 0.52, 0.59]

    a_ci_vals = [
        (0.13, 0.18),
        (0.18, 0.19),
        (0.15, 0.16),
        (0.14, 0.16),
        (0.21, 0.26),
        (0.28, 0.33),
    ]
    alpha_ci_vals = [
        (0.71, 0.75),
        (0.79, 0.83),
        (0.94, 0.98),
        (0.99, 1.03),
        (0.79, 0.83),
        (0.78, 0.82),
    ]
    beta_ci_vals = [
        (0.34, 0.38),
        (0.53, 0.55),
        (0.57, 0.59),
        (0.50, 0.51),
        (0.49, 0.50),
        (0.51, 0.52),
    ]
    g_ci_vals = [
        (0.22, 0.24),
        (0.32, 0.35),
        (0.39, 0.41),
        (0.43, 0.46),
        (0.50, 0.53),
        (0.57, 0.61),
    ]

    if fixed_temperature in temps_all:
        temp_idx = temps_all.index(fixed_temperature)

        ci_vals = np.array(
            [
                a_ci_vals[temp_idx],
                alpha_ci_vals[temp_idx],
                beta_ci_vals[temp_idx],
                g_ci_vals[temp_idx],
            ]
        )

        return (
            {
                "a_fun": lambda temp: a_vals[temp_idx],
                "c_fun": lambda temp: c_vals[temp_idx],
                "alpha_fun": lambda temp: alpha_vals[temp_idx],
                "beta_fun": lambda temp: beta_vals[temp_idx],
                "g_fun": lambda temp: g_vals[temp_idx],
            },
            ci_vals,
        )

    raise ValueError("fixed temperature is not in temps_list - did you make a typo?")


def build_param_interpolators(max_min_values: None | np.ndarray = None) -> dict:
    """
    Build interpolation functions for temperature-dependent parameters.
    Returns a dict of callables: a_fun, c_fun, alpha_fun, beta_fun, g_fun.
    """
    f = 1.5
    temps_all = np.array([20, 22, 24, 26, 28, 30])
    a_vals = np.arange(0.23, 0.28 + 1e-9, 0.01) / f
    c_vals = np.arange(0.040, 0.065 + 1e-9, 0.005) / f
    alpha_vals = np.arange(0.81, 0.86 + 1e-9, 0.01)
    beta_vals = np.arange(0.46, 0.56 + 1e-9, 0.02)
    g_vals = np.arange(0.27, 0.42 + 1e-9, 0.03) / f

    if max_min_values is not None:
        a_vals = np.arange(max_min_values[0, 0], max_min_values[1, 0] + 1e-9, 0.01) / f
        c_vals = np.arange(max_min_values[0, 1], max_min_values[1, 1] + 1e-9, 0.005) / f
        alpha_vals = np.arange(max_min_values[0, 2], max_min_values[1, 2] + 1e-9, 0.01)
        beta_vals = np.arange(max_min_values[0, 3], max_min_values[1, 3] + 1e-9, 0.02)
        g_vals = np.arange(max_min_values[0, 4], max_min_values[1, 4] + 1e-9, 0.03) / f

    return {
        "a_fun": make_interp_fun(temps_all, a_vals),
        "c_fun": make_interp_fun(temps_all, c_vals),
        "alpha_fun": make_interp_fun(temps_all, alpha_vals),
        "beta_fun": make_interp_fun(temps_all, beta_vals),
        "g_fun": make_interp_fun(temps_all, g_vals),
    }


def make_interp_fun(x, y):
    """Return a scalar interpolation function for arrays x,y."""
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    def f_scalar(t):
        return float(np.interp(t, x_arr, y_arr))

    return f_scalar


def survival_rates(stage, temperature):
    """Return survival probability for life stage at a given temperature."""
    egg_stage = 1
    pupae_stage = 3

    if stage == egg_stage:
        return 0.9 if (20 <= temperature <= 30) else 0.0
    if stage == pupae_stage:
        return 1.0
    return 1.0


def build_pars_for_larva(larvae_df_row, temperature, param_funs, cs):
    """
    Build the pars dict used by rdg_model for the larva.
    - larvae_df_row: a pandas Series (one row) representing a larva (must include *_fac and FA_actual/W).
    - temperature: current temperature scalar
    - param_funs: interpolator functions (a_fun, c_fun, alpha_fun, beta_fun)
    - cs: constants dict (must include Rth and sigma)
    Returns: pars dict with keys a,c,alpha,beta,g,Rth,sigma
    """
    # param factors (if provided) are used multiplicatively
    if larvae_df_row.shape[0] > 0:
        param_rows = larvae_df_row.loc[:, "a_fac":"g_fac"]
        a_fac = param_rows.iloc[:, 0]
        alpha_fac = param_rows.iloc[:, 1]
        beta_fac = param_rows.iloc[:, 2]
        g_fac = param_rows.iloc[:, 3]
        a_val = param_funs["a_fun"](temperature) * a_fac
        c_val = param_funs["c_fun"](temperature)
        alpha_val = param_funs["alpha_fun"](temperature) * alpha_fac
        beta_val = param_funs["beta_fun"](temperature) * beta_fac
        W = larvae_df_row["W"]
        g_val = param_funs["g_fun"](temperature) * g_fac

        return np.array(
            [
                a_val,
                np.full_like(W, c_val),
                alpha_val,
                beta_val,
                g_val,
                np.full_like(W, cs["Rth"]),
                np.full_like(W, cs["sigma"]),
            ]
        ).T
