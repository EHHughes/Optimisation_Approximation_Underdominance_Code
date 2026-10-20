Index for this folder (see top-level README.md for folder description)

**Files**
- agent_based_larval_model.py: efficient Python implementation of Ziqian's model, designed to run quickly and written in a way that it is easy to modify and debug
- input_output_pairs.py: code to generate input-output pairs including genetics
- sync_rng_random_functions.py: SyncRNG replacements for np.random functions used to cross-check R and Python implementations
- sync_rng_random_functions.r: SyncRNG replacements for R random functions used to cross-check R and Python implementations
- IBM_Model_6.4.Rmd: Ziqian's original R code (with SyncRNG checks included)

**Directories**
- helpers: helper modules broken out from the main IBM file to keep it manageable:
  - model_config.py — temperature-dependent parameter tables and interpolators; `build_constant_functions`, `build_param_interpolators`, `build_pars_for_larva`, `survival_rates`.
  - feeding_and_movement.py — per-timestep larval dynamics: `update_temperature`, `movement`, `apply_feeding_interference`, `apply_feeding_exploitation`, `rdg_model`, `state_update`.
  - mating.py — Mendelian random-mating and gene-drive conversion: `_mate_adults`, `get_genotype_distribution_for_stage_pop`, `update_larvae_genotypes`, `update_stage_pop_with_genotypes_pupae_and_adults`.
  - life_history.py — discrete life-stage transitions: `remove_dead_larvae`, `commit_to_pupation`, `transfer_pupated`.
  - logging.py — hourly and per-cycle statistics collection: `collect_larval_stats`, `append_hourly_row`, `append_cycle_stats_row`.
- tests: tests for the agent-based model and input-output pair generation
- legacy: old code versions predating the finalised implementation from Ziqian