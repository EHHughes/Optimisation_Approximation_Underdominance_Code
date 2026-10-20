"""helpers — submodules split from agent_based_larval_model_helpers.

Submodules
----------
mating
    Random mating, Mendelian inheritance, gene-drive conversion, and
    per-larva genotype assignment.
model_config
    Temperature-dependent parameter interpolators, survival rates, and
    per-larva parameter construction.
feeding_and_movement
    Per-timestep in-model behaviour: temperature perturbation, spatial
    movement, interference / exploitation feeding competition, and the
    RDG inner ODE with its numerical integrator.
logging
    Per-hour and per-cycle bookkeeping: stat collection and time-series
    row appending for larval_time_series and pupation_stats.
life_history
    Per-timestep life-stage transitions: larval mortality, pupation
    commitment, and transfer of pupated individuals.
"""

from IBM_Implementations.helpers.mating import (
    _mate_adults,
    get_genotype_distribution_for_stage_pop,
    update_larvae_genotypes,
    update_stage_pop_with_genotypes_pupae_and_adults,
)
from IBM_Implementations.helpers.model_config import (
    make_interp_fun,
    build_param_interpolators,
    build_constant_functions,
    survival_rates,
    build_pars_for_larva,
)
from IBM_Implementations.helpers.feeding_and_movement import (
    update_temperature,
    movement,
    apply_feeding_interference,
    apply_feeding_exploitation,
    rdg_model,
    state_update,
)
from IBM_Implementations.helpers.logging import (
    collect_larval_stats,
    append_hourly_row,
    append_cycle_stats_row,
)
from IBM_Implementations.helpers.life_history import (
    remove_dead_larvae,
    commit_to_pupation,
    transfer_pupated,
)

__all__ = [
    # mating
    "_mate_adults",
    "get_genotype_distribution_for_stage_pop",
    "update_larvae_genotypes",
    "update_stage_pop_with_genotypes_pupae_and_adults",
    # model_config
    "make_interp_fun",
    "build_param_interpolators",
    "build_constant_functions",
    "survival_rates",
    "build_pars_for_larva",
    # feeding_and_movement
    "update_temperature",
    "movement",
    "apply_feeding_interference",
    "apply_feeding_exploitation",
    "rdg_model",
    "state_update",
    # logging
    "collect_larval_stats",
    "append_hourly_row",
    "append_cycle_stats_row",
    # life_history
    "remove_dead_larvae",
    "commit_to_pupation",
    "transfer_pupated",
]
