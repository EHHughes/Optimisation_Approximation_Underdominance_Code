"""Code for an agent based model which reimplements Ziqian's model, now extended to account for genetics."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from paths import IBM_OUTPUT_DIR
from SyncRNG import SyncRNG

from IBM_Implementations.helpers.feeding_and_movement import (
    apply_feeding_exploitation,
    apply_feeding_interference,
    movement,
    rdg_model,
    state_update,
    update_temperature,
)
from IBM_Implementations.helpers.life_history import (
    commit_to_pupation,
    remove_dead_larvae,
    transfer_pupated,
)
from IBM_Implementations.helpers.logging import (
    append_cycle_stats_row,
    append_hourly_row,
    collect_larval_stats,
)
from IBM_Implementations.helpers.mating import (
    get_genotype_distribution_for_stage_pop,
    update_larvae_genotypes,
    update_stage_pop_with_genotypes_pupae_and_adults,
)
from IBM_Implementations.helpers.model_config import (
    build_constant_functions,
    build_param_interpolators,
    build_pars_for_larva,
    survival_rates,
)
from IBM_Implementations.sync_rng_random_functions import (
    sync_rng_binomial,
    sync_rng_choice,
    sync_rng_uniform,
)


def run_simulation(
    seed: int = 1234,
    cycles: None | int = None,
    map_size: int | None = None,
    parameter_ranges: np.ndarray | None = None,
    random_mating: bool = True,
    fixed_temperature: int | None = None,
    feeding_type: str = "interference",
    sync_rng: bool = False,
    genotype_penalties: dict | None = None,
    prop_eggs_genetically_modified: float | None = None,
    conversion_rate: float | None = None,
    eggs_count: int | None = None,
    record_larvae: bool = False,
) -> dict:
    """Standard interface to the model. Runs simulations with standard constants and returns the full state dict.

    **Parameters**
    1. seed: int
        - an integer value for the numpy.random.seed (if sync_rng is false) or SyncRNG seed.
    2. cycles: int
        - the number of cycles to run the model for
    3. map_size: int | None = None
        - optionally override the standard map size
    4. parameter_ranges: np.ndarray | None = None
        - parameter ranges to sample over (if overwriting standard parameter values). Currently only implemented for the case where temperature varies over time
    5. random_mating: bool = True
        - if True incorporates genetics and random mating. If False follows the same logic as Ziqian's codebase
    6. fixed_temperature: int | None = None
        - a fixed temperature value to run simulations at (if None temperature follows a random walk)
    7. feeding_type: str = "interference"
        - either "interference" or "exploitative" to set the feeding type used in the IBM
    8. sync_rng: bool = False
        - if True use SyncRNG to produce code which can be tested against R code. If False use standard numpy.random implementations
    9. genotype_penalties: dict | None = None
        - per-genotype, per-parameter multiplicative penalties, e.g.
          {"WW": {"a_fac": 1.0, ...}, "WG": {...}, "GG": {...}}. If None the
          defaults from initialize_simulation() are used.
    10. prop_eggs_genetically_modified: float | None = None
        - overrides the fraction of initial eggs that are GG
    11. conversion_rate: float | None = None
        - overrides the heterozygote→GG gene-drive conversion rate
    12. eggs_count: int | None = None
        - overrides the number of eggs entering the first cycle (otherwise the
          default of 100 from initialize_simulation is used)
    13. record_larvae: bool = False
        - if True, record a per-hour, per-larva log of (cycle, hour, larva_id,
          W, R, x, y) into state["larva_trajectories"] (a DataFrame). Off by
          default so normal runs are unaffected; used only for figure/diagnostic
          purposes as it is memory-heavy for large maps.

    **Returns**
    - state: a dict containing the evolution of the system over time

    """
    # build parameter interpolators and initial state
    if sync_rng:
        rng_generator = SyncRNG(seed)
    else:
        np.random.seed(seed)
        rng_generator = None
    init_kwargs = {"cycles": cycles}
    if map_size is not None:
        init_kwargs["map_size"] = map_size
    state = initialize_simulation(**init_kwargs)

    if feeding_type not in ["interference", "exploitative"]:
        raise ValueError("feeding_type must be either 'interference' or 'exploitative'")
    state["constants"]["feeding_type"] = feeding_type

    if cycles is not None:
        state["constants"]["cycles"] = cycles

    if genotype_penalties is not None:
        state["constants"]["genotype_penalties"] = genotype_penalties

    if prop_eggs_genetically_modified is not None:
        state["constants"]["prop_eggs_genetically_modified"] = (
            prop_eggs_genetically_modified
        )

    if conversion_rate is not None:
        state["constants"]["conversion_rate"] = conversion_rate

    if eggs_count is not None:
        state["constants"]["eggs_count"] = int(eggs_count)

    if fixed_temperature is None:
        param_funs = build_param_interpolators(parameter_ranges)
        ci_vals_array = None
    else:
        param_funs, ci_vals_array = build_constant_functions(fixed_temperature)

    if record_larvae:
        state["larva_trajectories"] = []

    for cycle in range(state["constants"]["cycles"]):
        state = run_cycle(
            cycle,
            state,
            param_funs,
            ci_vals_array,
            random_mating=random_mating,
            random_generator=rng_generator,
            fixed_temperature=(fixed_temperature is not None),
            record_larvae=record_larvae,
        )

    state["larval_time_series"] = pd.DataFrame(state["larval_time_series"])
    state["pupation_stats"] = pd.DataFrame(state["pupation_stats"])
    if record_larvae:
        state["larva_trajectories"] = (
            pd.concat(state["larva_trajectories"], ignore_index=True)
            if state["larva_trajectories"]
            else pd.DataFrame(columns=["cycle", "hour", "larva_id", "W", "R", "x", "y"])
        )
    return state


def initialize_simulation(cycles: int | None = None, map_size: int = 20) -> dict:
    """Return default constants, empty data structures and RNG setup."""
    constants = {
        "map_size": map_size,
        "time_steps": 300,
        "cycles": 50,
        "dispersal": 0.10,
        "Rth": 15,
        "sigma": 23,
        "Rpup": 87.9,
        "W0": 4.1,
        "R0": 0.4 * 4.1,
        "n_winners": 4,
        "initial_food_value": 2700,
        "initial_food": None,
        "refill_value": 75,
        "refill_amount": None,
        "temperature": 20.0,
        "pup_delay": 37.5,
        "density_level": 1.4,
        "V_cell": 5,
        "feeding_type": "interference",
        "assimilation_efficacy": 0.25,
        "eggs_per_female": 30,
        "pond_carrying_capacity": np.inf,
        "eggs_count": 100,
        "prop_eggs_genetically_modified": 0.75,
        "prop_eggs_heterozygotes": 0,
        "param_sds": {"a": 0.10, "alpha": 0.024, "beta": 0.022, "g": 0.035},
        "genotype_penalties": {
            "WW": {"a_fac": 1.0, "alpha_fac": 1.0, "beta_fac": 1.0, "g_fac": 1.0},
            "WG": {
                "a_fac": 1.0,
                "alpha_fac": 0.4195921036285358,
                "beta_fac": 1.0,
                "g_fac": 0.1361132399292676,
            },
            "GG": {
                "a_fac": 0.6385113095508155,
                "alpha_fac": 0.8621691041533351,
                "beta_fac": 0.2531671406738747,
                "g_fac": 0.5672969157286821,
            },
        },
        "conversion_rate": 0.0,
    }

    if constants["pond_carrying_capacity"] is np.inf:
        # follow logic in Ziqian's final version to set carrying capacity if not specified
        constants["pond_carrying_capacity"] = int(
            constants["V_cell"]
            * constants["density_level"]
            * constants["map_size"] ** 2
        )
    if constants["feeding_type"] == "interference":
        constants["initial_food"] = (
            constants["initial_food_value"] * constants["n_winners"]
        )
        constants["refill_amount"] = constants["refill_value"] * constants["n_winners"]
    if constants["feeding_type"] == "exploitative":
        constants["initial_food"] = (
            constants["initial_food_value"]
            * constants["density_level"]
            * constants["V_cell"]
        )
        constants["refill_amount"] = (
            constants["refill_value"] * constants["density_level"] * constants["V_cell"]
        )

    if cycles is not None:
        constants["cycles"] = cycles
    # Pre-allocate containers
    number_of_cycles = int(constants["cycles"])
    number_of_time_steps = int(constants["time_steps"])
    stage_pop = pd.DataFrame(
        index=range(number_of_cycles),
        columns=["Eggs", "Larvae", "Pupae", "Adults"],
        dtype=int,
    )
    stage_pop_including_genotypes = pd.DataFrame(
        index=range(number_of_cycles),
        columns=[
            stage_pop + "_" + genotype
            for genotype in ["WW", "WG", "GG"]
            for stage_pop in ["Eggs", "Larvae", "Pupae", "Adults"]
        ],
        dtype=int,
    )
    pupated_all = [None] * number_of_cycles
    deceased_all = [None] * number_of_cycles
    temperature_matrix = np.full((number_of_cycles, number_of_time_steps), np.nan)
    food_matrix = np.full((number_of_cycles, number_of_time_steps), np.nan)
    W_matrix = np.full((number_of_cycles, number_of_time_steps), np.nan)
    R_matrix = np.full((number_of_cycles, number_of_time_steps), np.nan)

    pupation_stats: list[dict] = []

    larval_time_series: list[dict] = []

    return {
        "constants": constants,
        "stage_pop": stage_pop,
        "stage_pop_including_genotypes": stage_pop_including_genotypes,
        "pupated_all": pupated_all,
        "deceased_all": deceased_all,
        "temperature_matrix": temperature_matrix,
        "food_matrix": food_matrix,
        "W_matrix": W_matrix,
        "R_matrix": R_matrix,
        "pupation_stats": pupation_stats,
        "larval_time_series": larval_time_series,
    }


def run_cycle(
    cycle,
    state,
    param_funs,
    ci_vals_array: np.ndarray | None = None,
    random_mating: bool = True,
    fixed_temperature: bool = False,
    random_generator: SyncRNG | None = None,
    record_larvae: bool = False,
):
    """
    Run a single cycle (life-cycle) of the IBM + RDG model.
    cycle: cycle index
    state: dict that contains the objects returned by initialize_simulation()
    param_funs: dict containing a_fun, c_fun, alpha_fun, beta_fun, g_fun
    Returns: updated state dict
    """
    # alias constants
    cs = state["constants"]
    map_size = cs["map_size"]
    time_steps = cs["time_steps"]
    Rth = cs["Rth"]
    sigma = cs["sigma"]
    Rpup = cs["Rpup"]
    W0 = cs["W0"]
    R0 = cs["R0"]
    initial_food = cs["initial_food"]
    refill_amount = cs["refill_amount"]
    temperature = cs["temperature"]
    eggs_count = cs["eggs_count"]
    pup_delay = cs["pup_delay"]
    eggs_per_female = cs["eggs_per_female"]
    pond_egg_carrying_capacity = cs["pond_carrying_capacity"]

    feeding_type = cs["feeding_type"]
    assimilation_efficacy = cs["assimilation_efficacy"]
    genotype_penalties = cs["genotype_penalties"]
    conversion_rate = cs["conversion_rate"]

    cycle_start = cycle * (time_steps + 10)
    grid_food = np.full((map_size, map_size), initial_food, dtype=float)

    # Stage population start values
    larvae_survived = initialise_cycle_stage_pop(
        state, cycle, eggs_count, temperature, cs, random_mating
    )

    larvae_df = initialise_larvae_dataframe(
        larvae_survived=larvae_survived,
        map_size=map_size,
        W0=W0,
        R0=R0,
        param_funs=param_funs,
        temperature=temperature,
        ci_vals_array=ci_vals_array,
        param_sds=cs["param_sds"],
        random_generator=random_generator,
        random_mating=random_mating,
        stage_pop_including_genotypes=state["stage_pop_including_genotypes"],
        cycle=cycle,
        genotype_penalties=genotype_penalties,
    )

    pupae_count = 0
    temperature_record = np.full(time_steps, np.nan)
    prev_larvae_count = None

    for hour in range(1, time_steps + 1):
        t_global = cycle_start + hour

        if not fixed_temperature:
            temperature = update_temperature(temperature)
            temperature = min(max(temperature, 20.0), 30.0)
        temperature_record[hour - 1] = temperature

        if hour % 24 == 0:
            grid_food = grid_food + refill_amount
        state["food_matrix"][cycle, hour - 1] = grid_food.sum()

        current_larvae = larvae_df.shape[0]
        if current_larvae > 0:
            larvae_df = movement(larvae_df, map_size, random_generator)

            hour_stats = collect_larval_stats(
                larvae_df, current_larvae, grid_food, map_size
            )

            larvae_df["FA_actual"] = 0.0
            if feeding_type == "exploitative":
                larvae_df, grid_food = apply_feeding_exploitation(
                    larvae_df,
                    grid_food,
                    temperature,
                    param_funs,
                    Rth,
                    sigma,
                    assimilation_efficacy,
                )
            elif feeding_type == "interference":
                larvae_df, grid_food = apply_feeding_interference(
                    larvae_df,
                    grid_food,
                    temperature,
                    param_funs,
                    Rth,
                    sigma,
                    assimilation_efficacy,
                    max_competitors=cs["n_winners"],
                    random_generator=random_generator,
                )
            else:
                raise ValueError(
                    "feeding type must be either exploitative or interference"
                )

            # compute new weights and reserves for all larvae
            state_vec = larvae_df[["W", "R"]].values
            pars = build_pars_for_larva(larvae_df, temperature, param_funs, cs)
            new_states = state_update(
                state_vec, pars, rdg_model, larvae_df.loc[:, "FA_actual"]
            )
            larvae_df.loc[:, "W"] = new_states[:, 0]
            larvae_df.loc[:, "R"] = new_states[:, 1]

            if record_larvae:
                # Snapshot every living larva's state *before* this hour's
                # death/pupation removal, so the trajectory captures the reserve
                # crossing R<0 (starvation) or R>=Rpup (pupation).
                snapshot = larvae_df[["larva_id", "W", "R", "x", "y"]].copy()
                snapshot.insert(0, "hour", hour)
                snapshot.insert(0, "cycle", cycle)
                state["larva_trajectories"].append(snapshot)
        else:
            hour_stats = collect_larval_stats(
                larvae_df, current_larvae, grid_food, map_size
            )

        larvae_df, death_count = remove_dead_larvae(larvae_df, state, cycle)

        # commit larvae to pupation which exceed the threshold Rpup
        larvae_df, new_commitments = commit_to_pupation(
            larvae_df, t_global, Rpup, pup_delay
        )

        # Pupation: move to pupated list
        larvae_df, pupated_now, pupae_count = transfer_pupated(
            larvae_df, state, cycle, t_global, pupae_count
        )

        # store means for W,R
        state["W_matrix"][cycle, hour - 1] = (
            larvae_df["W"].mean() if larvae_df.shape[0] > 0 else np.nan
        )
        state["R_matrix"][cycle, hour - 1] = (
            larvae_df["R"].mean() if larvae_df.shape[0] > 0 else np.nan
        )

        if prev_larvae_count is not None and prev_larvae_count > 0:
            survival_rate = current_larvae / prev_larvae_count
        else:
            survival_rate = np.nan
        prev_larvae_count = current_larvae

        # Append summary row to larval_time_series
        append_hourly_row(
            state,
            cycle,
            hour,
            current_larvae,
            hour_stats,
            survival_rate,
            new_commitments,
            pupated_now,
            death_count,
        )

    # finalize cycle-level stats
    total_deaths = (
        state["deceased_all"][cycle].shape[0]
        if state["deceased_all"][cycle] is not None
        else 0
    )
    remaining_larvae = larvae_df.shape[0]

    state["stage_pop"].at[cycle, "Pupae"] = pupae_count
    if random_mating:
        state["stage_pop_including_genotypes"] = (
            update_stage_pop_with_genotypes_pupae_and_adults(state, cycle, temperature)
        )
    adults_count = int(np.floor(pupae_count * survival_rates(3, temperature)))
    state["stage_pop"].at[cycle, "Adults"] = adults_count

    if random_mating and cycle <= cs["cycles"] - 2:
        state["stage_pop_including_genotypes"].loc[cycle + 1, :] = (
            get_genotype_distribution_for_stage_pop(
                state["stage_pop_including_genotypes"],
                cycle + 1,
                random_mating=random_mating,
                eggs_per_female=cs["eggs_per_female"],
                heterozygote_homozygote_conversion_rate=conversion_rate,
                pond_carrying_capacity=pond_egg_carrying_capacity,
            )
        )
        cs["eggs_count"] = np.sum(
            state["stage_pop_including_genotypes"].loc[cycle + 1, :]
        )
    else:
        females_count = np.floor(adults_count / 2)
        if females_count > 0:
            probability_of_retention = min(
                1, pond_egg_carrying_capacity / (females_count * eggs_per_female)
            )
            if random_generator is not None:
                retained_females = sync_rng_binomial(
                    int(females_count), probability_of_retention, s=random_generator
                )
            else:
                retained_females = np.random.binomial(
                    females_count, probability_of_retention
                )
        else:
            retained_females = 0
        if retained_females == 0 and random_generator is not None and females_count > 0:
            retained_females = 1
        eggs_count = retained_females * eggs_per_female
        cs["eggs_count"] = eggs_count
    state["temperature_matrix"][cycle, :] = temperature_record

    append_cycle_stats_row(state, cycle, larvae_df, cycle_start, grid_food)

    # set final temperature
    state["constants"]["temperature"] = temperature

    # also set eggs_count for the next cycle (useful for the input-output pair function)
    state["constants"]["eggs_count"] = cs["eggs_count"]

    # return updated state (we mutate in-place but also return for convenience)
    return state


def initialise_cycle_stage_pop(
    state: dict,
    cycle: int,
    eggs_count: int,
    temperature: float,
    cs: dict,
    random_mating: bool,
) -> int:
    """Set start-of-cycle egg and larva counts in state and return larvae_survived.

    For cycle 0 with random_mating, splits the egg cohort across WW / WG / GG
    genotypes according to the proportions in cs.  For all other cases the
    total egg count is recorded directly.  Egg→larva survival is then applied
    to produce the initial larval cohort size.

    Parameters
    ----------
    state:
        Simulation state dict (``stage_pop`` and
        ``stage_pop_including_genotypes`` are mutated in-place).
    cycle:
        Zero-based cycle index.
    eggs_count:
        Total number of eggs entering this cycle.
    temperature:
        Current temperature (used for egg→larva survival rate lookup).
    cs:
        Constants dict (reads ``prop_eggs_genetically_modified``,
        ``prop_eggs_heterozygotes``).
    random_mating:
        Whether genetics are tracked.

    Returns
    -------
    larvae_survived:
        Number of larvae alive at the start of the cycle.
    """
    if random_mating and cycle == 0:
        state["stage_pop_including_genotypes"].iloc[0, 0] = int(
            (1 - cs["prop_eggs_genetically_modified"] - cs["prop_eggs_heterozygotes"])
            * eggs_count
        )
        state["stage_pop_including_genotypes"].iloc[0, 4] = int(
            cs["prop_eggs_heterozygotes"] * eggs_count
        )
        state["stage_pop_including_genotypes"].iloc[0, 8] = int(
            cs["prop_eggs_genetically_modified"] * eggs_count
        )

        state["stage_pop"].at[cycle, "Eggs"] = int(
            state["stage_pop_including_genotypes"].iloc[0, 8]
            + state["stage_pop_including_genotypes"].iloc[0, 4]
            + state["stage_pop_including_genotypes"].iloc[0, 0]
        )
        if cs["prop_eggs_genetically_modified"] + cs["prop_eggs_heterozygotes"] > 1:
            raise ValueError(
                "The proportion of eggs which are not WW must be less than 1!. "
                "Check prop_eggs* to make sure they are set to plausible values"
            )
    else:
        state["stage_pop"].at[cycle, "Eggs"] = eggs_count

    if random_mating:
        state["stage_pop_including_genotypes"].iloc[cycle, 1] = np.floor(
            state["stage_pop_including_genotypes"].iloc[cycle, 0]
            * survival_rates(1, temperature)
        )
        state["stage_pop_including_genotypes"].iloc[cycle, 5] = np.floor(
            state["stage_pop_including_genotypes"].iloc[cycle, 4]
            * survival_rates(1, temperature)
        )
        state["stage_pop_including_genotypes"].iloc[cycle, 9] = np.floor(
            state["stage_pop_including_genotypes"].iloc[cycle, 8]
            * survival_rates(1, temperature)
        )
        larvae_survived = int(
            state["stage_pop_including_genotypes"].loc[cycle, "Larvae_WW"]
            + state["stage_pop_including_genotypes"].loc[cycle, "Larvae_WG"]
            + state["stage_pop_including_genotypes"].loc[cycle, "Larvae_GG"]
        )
    else:
        larvae_survived = int(np.floor(eggs_count * survival_rates(1, temperature)))

    state["stage_pop"].at[cycle, "Larvae"] = larvae_survived
    return larvae_survived


def initialise_larvae_dataframe(
    larvae_survived: int,
    map_size: int,
    W0: float,
    R0: float,
    param_funs: dict,
    temperature: float,
    ci_vals_array: np.ndarray | None,
    param_sds: dict,
    random_generator,
    random_mating: bool,
    stage_pop_including_genotypes,
    cycle: int,
    genotype_penalties: dict,
) -> pd.DataFrame:
    """Create and return the initial larvae DataFrame for a single cycle.

    Assigns random spatial positions, base physiological state (W, R), and
    per-larva parameter uncertainty factors.  If random_mating is True,
    genotypes are also drawn and their penalties applied.

    Returns an empty DataFrame (with the standard columns) when
    larvae_survived is zero.
    """
    if larvae_survived == 0:
        return pd.DataFrame(columns=["larva_id", "W", "R", "x", "y", "pupate_at"])

    grid_positions = np.arange(1, map_size + 1)
    if random_generator is not None:
        x_positions = sync_rng_choice(
            grid_positions, s=random_generator, size=larvae_survived, replace=True
        )
        y_positions = sync_rng_choice(
            grid_positions, s=random_generator, size=larvae_survived, replace=True
        )
    else:
        x_positions = np.random.choice(
            grid_positions, size=larvae_survived, replace=True
        )
        y_positions = np.random.choice(
            grid_positions, size=larvae_survived, replace=True
        )

    larvae_df = pd.DataFrame(
        {
            # Stable per-larva identifier, unique within a cycle.  Carried as a
            # column (never the index) so it survives the reset_index(drop=True)
            # calls in remove_dead_larvae / initialise, letting a larva be
            # followed across hours and joined back to its fate.
            "larva_id": np.arange(larvae_survived, dtype=int),
            "W": np.full(larvae_survived, W0, dtype=float),
            "R": np.full(larvae_survived, R0, dtype=float),
            "x": x_positions,
            "y": y_positions,
            "pupate_at": np.full(larvae_survived, np.nan),
            "a_fac": np.full(larvae_survived, 1.0),
            "alpha_fac": np.full(larvae_survived, 1.0),
            "beta_fac": np.full(larvae_survived, 1.0),
            "g_fac": np.full(larvae_survived, 1.0),
            "genotype": np.full(larvae_survived, "WW"),
        }
    )

    param_func_names = ["a_fun", "alpha_fun", "beta_fun", "g_fun"]
    mean_values = np.array(
        [param_funs.get(fun_name)(temperature) for fun_name in param_func_names]
    )
    if np.any(mean_values == 0):
        raise ValueError(
            "Mean parameter value is zero, cannot apply uncertainty factors."
        )

    scaled_sds = np.array(list(param_sds.values())) / 1.96
    if ci_vals_array is None:
        larvae_df.loc[:, "a_fac":"g_fac"] = np.random.normal(
            loc=1.0,
            scale=scaled_sds,
            size=(larvae_survived, len(param_func_names)),
        )
    else:
        if random_generator is None:
            parameter_factor_arr = np.random.uniform(
                ci_vals_array[:, 0],
                ci_vals_array[:, 1],
                size=(larvae_survived, len(param_func_names)),
            )
            larvae_df.loc[:, "a_fac":"g_fac"] = parameter_factor_arr / mean_values
        else:
            larvae_df.loc[:, "a_fac":"g_fac"] = (
                sync_rng_uniform(
                    ci_vals_array[:, 0],
                    ci_vals_array[:, 1],
                    size=(larvae_survived, len(param_func_names)),
                    s=random_generator,
                )
                / mean_values
            )

    if random_mating:
        larvae_df = update_larvae_genotypes(
            larvae_df,
            stage_pop_including_genotypes,
            cycle,
            genotype_penalties,
        )

    return larvae_df


def plot_stage_summary(state, plot_genotypes: bool = False) -> None:
    """Plot start-of-stage populations across cycles."""
    stage_df = state["stage_pop"].reset_index(drop=True).copy()
    stage_df["Cycle"] = np.arange(1, stage_df.shape[0] + 1)
    long_df = pd.melt(
        stage_df,
        id_vars=["Cycle"],
        value_vars=["Eggs", "Larvae", "Pupae", "Adults"],
        var_name="Stage",
        value_name="Count",
    )
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=long_df, x="Cycle", y="Count", hue="Stage", marker="o")
    plt.tight_layout()
    plt.show()

    if plot_genotypes:
        stage_df = state["stage_pop_including_genotypes"].reset_index(drop=True).copy()
        stage_df["Cycle"] = np.arange(1, stage_df.shape[0] + 1)
        long_df = pd.melt(
            stage_df,
            id_vars=["Cycle"],
            value_vars=[
                stage_pop + "_" + genotype
                for genotype in ["WW", "WG", "GG"]
                for stage_pop in ["Eggs", "Larvae", "Pupae", "Adults"]
            ],
            var_name="Stage",
            value_name="Count",
        )
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=long_df, x="Cycle", y="Count", hue="Stage", marker="o")
        plt.tight_layout()
        plt.show()


def export_results(
    state,
    pupation_stats_path=str(
        IBM_OUTPUT_DIR / "pupation_stats_optimised_python_version.csv"
    ),
    larval_ts_path=str(
        IBM_OUTPUT_DIR / "larval_time_series_optimisied_python_version.csv"
    ),
):
    """Export pupation_stats and larval_time_series to CSV files."""
    state["pupation_stats"].to_csv(pupation_stats_path, index=False)
    state["larval_time_series"].to_csv(larval_ts_path, index=False)
    print(f"Wrote {pupation_stats_path} and {larval_ts_path}")
