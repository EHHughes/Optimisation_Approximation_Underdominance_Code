"""Gene-drive releases into a single, equilibrated wild-type population.

The optimisation objectives (``gene_drive_objective.py``) score a drive from a
founding cohort of eggs whose GG fraction is set by hand.  That is a clean
score but an unrealistic release: a real drive is released into a population
that is already sitting at its carrying capacity, and it has to make headway
against a resident wild type rather than against a cohort assembled around it.

This module runs that second experiment for one location.  A wild-type
population is left to settle to equilibrium, GG adults are then released at a
range of frequencies, and every stage count the IBM produces is recorded so the
outcome can be read at the level of the mechanism -- which genotype's larvae
starved, which timed out, and which reached adulthood -- rather than only as a
final allele frequency.

**Protocol**

1. Seed the location with ``initial_ww`` wild-type adults and run
   ``n_burn_in_generations`` conversion-free generations (:func:`run_burn_in`).
   The mean of the last few is reported as the equilibrium abundance.
2. Add GG adults to that equilibrium state at each requested release frequency
   (:func:`apply_release`).
3. Advance each release ``n_generations`` further generations under the
   drive's own conversion rate and fitness penalties.

Replicates are *paired*: a replicate burns in once and every release frequency
branches from that same equilibrium state, so panels differ because of the
release rather than because of the wild-type background they landed in.  The
post-release branches of a replicate are handed the same seed as well, though
that buys less than it looks: the number of random draws a generation makes
depends on how many mosquitoes are in it, so the streams diverge as soon as the
releases differ in size.  The shared burn-in is what the pairing rests on.

**Cost.**  One generation is one IBM cycle, about 15 s at equilibrium
abundance, and there are ``n_replicates * (n_burn_in_generations +
n_frequencies * n_generations)`` of them.  Branches run in parallel across
``n_workers`` processes, but this is still minutes rather than seconds, which
is why :func:`simulate_single_location_releases` writes its raw counts to CSV
and the figures can be rebuilt from that file without re-simulating.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import multiprocess
from cpu_resources import available_cpu_count
from IBM_Implementations.input_output_pairs import (
    GENOTYPES,
    breed_adults_to_eggs,
    run_ibm_cycle_from_eggs,
)

#: Per-genotype counts recorded for every simulated generation.  ``starved``
#: and ``timed_out`` are the two ways a larva fails to become a pupa, and
#: ``larvae == starved + timed_out + pupae`` holds exactly.
STAGE_COLUMNS: tuple[str, ...] = (
    "eggs",
    "larvae",
    "starved",
    "timed_out",
    "pupae",
    "adults",
)

#: Egg carrying capacity of the pond, matching the value
#: ``Spatial_Simulations.one_d_simulations.agent_based_model`` hard-codes, so a
#: single-location run and a spatial run describe the same environment.
POND_CARRYING_CAPACITY: int = 2800

#: Phase markers on each row of the tidy output.  ``seed`` and ``release`` are
#: states rather than simulated cycles, so they carry adult counts only.
PHASE_SEED = "seed"
PHASE_BURN_IN = "burn_in"
PHASE_RELEASE = "release"
PHASE_POST_RELEASE = "post_release"

#: How many trailing burn-in generations the reported equilibrium averages over.
_EQUILIBRIUM_WINDOW = 3

#: Relative drift across the final burn-in generations above which the burn-in
#: is reported as unsettled.  The wild-type equilibrium is itself noisy (a
#: coefficient of variation near 0.2), so this only catches a population still
#: visibly trending, not ordinary generation-to-generation scatter.
_EQUILIBRIUM_TOLERANCE = 0.25


def run_generation(
    adults: np.ndarray,
    *,
    conversion_rate: float = 0.0,
    genotype_penalties: dict | None = None,
    eggs_per_female: int = 30,
    pond_carrying_capacity: int = POND_CARRYING_CAPACITY,
    temperature: int = 20,
    feeding_type: str = "interference",
    seed: int | None = None,
) -> dict[str, int]:
    """Advance one generation, returning every stage count by genotype.

    Composed exactly as ``one_d_simulations.agent_based_model`` composes a
    within-location generation -- adults are mated to eggs, the drive converts
    heterozygotes as those eggs are laid, and one IBM cycle carries them to
    adulthood -- so a single-location run and a spatial run are directly
    comparable.  The difference is only in what is kept: this returns the
    intermediate stages as well as the adults that come out.

    Conversion is applied at the adult -> egg step because a one-cycle IBM run
    never reaches the model's own between-cycle conversion; see
    ``run_ibm_from_eggs``.

    **Arguments**
    1. adults: np.ndarray
        - (3,) array of resident adults (WW, WG, GG)
    2. conversion_rate: float = 0.0
        - probability a WG egg is converted to GG as it is laid
    3. genotype_penalties: dict | None = None
        - per-genotype multiplicative parameter penalties; None uses the IBM's
          own defaults
    4. eggs_per_female: int = 30
    5. pond_carrying_capacity: int = POND_CARRYING_CAPACITY
    6. temperature: int = 20
        - fixed simulation temperature (must be one of 20, 22, 24, 26, 28, 30)
    7. feeding_type: str = "interference"
    8. seed: int | None = None
        - seed for this generation's draws.  Both the mating and the IBM draw
          from numpy's global RNG, so seeding here covers the whole generation.
          Pass an explicit seed to make a run repeatable.

    **Returns**
    - dict of ``{stage}_{genotype}`` counts, with ``Adults_*`` the population
      entering the next generation.  An extinct population returns all zeros.
    """
    if seed is not None:
        np.random.seed(seed)

    adults = np.asarray(adults)
    if adults.sum() <= 0:
        return {
            f"{stage}_{genotype}": 0
            for genotype in GENOTYPES
            for stage in ("Eggs", "Larvae", "Pupae", "Adults", "Starved", "TimedOut")
        }

    eggs = breed_adults_to_eggs(
        int(adults[0]),
        int(adults[1]),
        int(adults[2]),
        eggs_per_female=eggs_per_female,
        pond_carrying_capacity=pond_carrying_capacity,
        heterozygote_conversion_rate=conversion_rate,
    )

    stage_counts = run_ibm_cycle_from_eggs(
        eggs[0],
        eggs[1],
        eggs[2],
        temperature,
        feeding_type,
        genotype_penalties=genotype_penalties,
    )

    if stage_counts is None:
        # Every adult failed to breed: no eggs, so no cycle to run.
        return {
            f"{stage}_{genotype}": 0
            for genotype in GENOTYPES
            for stage in ("Eggs", "Larvae", "Pupae", "Adults", "Starved", "TimedOut")
        }

    return stage_counts


def adults_from_counts(stage_counts: dict[str, int]) -> np.ndarray:
    """Pull the (WW, WG, GG) adult vector out of a stage-count dict."""
    return np.array(
        [int(stage_counts[f"Adults_{genotype}"]) for genotype in GENOTYPES],
        dtype=np.int64,
    )


def apply_release(
    resident_adults: np.ndarray,
    release_frequency: float,
    mode: str = "additive",
) -> np.ndarray:
    """Release GG adults into a resident population at a given frequency.

    *release_frequency* is the GG proportion of the population immediately
    after the release, which is the ``prop_gg`` convention used by the
    gene-drive optimisation and by the spatial release comparisons.

    Two conventions are offered because they answer different questions:

    - ``additive`` (default) releases GG *on top of* the residents, so a
      location holding ``W`` of them receives ``W * f / (1 - f)`` GG adults.
      This is what a release programme actually does, and it perturbs total
      abundance above equilibrium.
    - ``replacement`` holds the total fixed and converts a share of the
      residents to GG.  Abundance is then unperturbed, which isolates the
      genetic effect of the release from the density-dependent response to
      adding mosquitoes.

    **Arguments**
    1. resident_adults: np.ndarray
        - (3,) array of resident adults (WW, WG, GG)
    2. release_frequency: float
        - GG proportion immediately after the release; must lie in [0, 1),
          since a frequency of 1 would need infinitely many released adults
    3. mode: str = "additive"

    **Returns**
    - (3,) array of adults immediately after the release
    """
    if not 0.0 <= release_frequency < 1.0:
        raise ValueError(
            f"release_frequency {release_frequency} is outside [0, 1); a "
            "frequency of 1 would require an unbounded number of released adults."
        )
    if mode not in ("additive", "replacement"):
        raise ValueError(
            f"mode must be 'additive' or 'replacement'; got {mode!r}."
        )

    released = np.asarray(resident_adults, dtype=np.int64).copy()
    total = int(released.sum())
    if total == 0:
        return released

    if mode == "additive":
        released[2] += int(round(total * release_frequency / (1.0 - release_frequency)))
        return released

    target_gg = int(round(total * release_frequency))
    # Take the replaced individuals from the residents in proportion to their
    # genotypes, so a replacement release changes the GG share without also
    # reshaping the rest of the population.
    remaining = total - target_gg
    resident_total = max(total - released[2], 1)
    scaled = released[:2] * remaining / resident_total
    replaced = np.floor(scaled).astype(np.int64)
    # Hand any rounding shortfall to the commonest resident genotype.
    shortfall = remaining - int(replaced.sum())
    if shortfall > 0 and released[:2].sum() > 0:
        replaced[int(np.argmax(released[:2]))] += shortfall
    return np.array([replaced[0], replaced[1], target_gg], dtype=np.int64)


def run_burn_in(
    initial_ww: float,
    n_generations: int,
    *,
    seed: int,
    eggs_per_female: int = 30,
    pond_carrying_capacity: int = POND_CARRYING_CAPACITY,
    temperature: int = 20,
    feeding_type: str = "interference",
) -> list[dict]:
    """Let a wild-type population settle, returning a record per generation.

    The burn-in carries no drive: penalties are irrelevant with only WW present
    and the conversion rate is zero, so this is the wild type finding its own
    density-dependent equilibrium.

    **Returns**
    - a list of ``n_generations`` stage-count dicts, each carrying a
      ``generation`` key.  Generations are numbered backwards from the release:
      the last entry is generation -1, the state the drive is released into.
    """
    records: list[dict] = []
    adults = np.array([int(initial_ww), 0, 0], dtype=np.int64)
    generation_seeds = np.random.SeedSequence(seed).generate_state(n_generations)

    for offset, generation_seed in enumerate(generation_seeds):
        stage_counts = run_generation(
            adults,
            conversion_rate=0.0,
            genotype_penalties=None,
            eggs_per_female=eggs_per_female,
            pond_carrying_capacity=pond_carrying_capacity,
            temperature=temperature,
            feeding_type=feeding_type,
            seed=int(generation_seed),
        )
        adults = adults_from_counts(stage_counts)
        records.append({**stage_counts, "generation": offset - n_generations})

    return records


def equilibrium_summary(burn_in_records: list[dict]) -> dict:
    """Summarise where a burn-in ended up, and whether it had stopped moving.

    The wild-type equilibrium in this model is noisy -- generation-to-generation
    swings of tens of per cent are ordinary -- so "settled" cannot mean a single
    converged number.  What is reported instead is the mean total abundance over
    the last few generations, and whether the trend across them is small
    relative to that mean.

    **Returns**
    - dict with ``mean``, ``minimum``, ``maximum``, ``drift`` (the relative
      change across the averaging window) and ``settled`` (drift within
      tolerance).
    """
    totals = np.array(
        [
            sum(record[f"Adults_{genotype}"] for genotype in GENOTYPES)
            for record in burn_in_records
        ],
        dtype=float,
    )
    window = totals[-_EQUILIBRIUM_WINDOW:] if totals.size else totals
    mean = float(window.mean()) if window.size else 0.0
    if window.size >= 2 and mean > 0:
        drift = float(abs(window[-1] - window[0]) / mean)
    else:
        drift = 0.0

    return {
        "mean": mean,
        "minimum": float(window.min()) if window.size else 0.0,
        "maximum": float(window.max()) if window.size else 0.0,
        "drift": drift,
        "settled": drift <= _EQUILIBRIUM_TOLERANCE,
    }


def _burn_in_worker(args: tuple) -> list[dict]:
    """Module-level so ``multiprocess`` can pickle it."""
    seed, kwargs = args
    return run_burn_in(seed=seed, **kwargs)


def _post_release_worker(args: tuple) -> list[dict]:
    """Advance one (replicate, frequency) branch from its released state.

    Module-level so ``multiprocess`` can pickle it.  Returns one stage-count
    record per generation, each tagged with its ``generation`` index.
    """
    released_adults, seed, n_generations, kwargs = args

    records: list[dict] = []
    adults = np.asarray(released_adults, dtype=np.int64)
    generation_seeds = np.random.SeedSequence(seed).generate_state(n_generations)

    for offset, generation_seed in enumerate(generation_seeds):
        stage_counts = run_generation(adults, seed=int(generation_seed), **kwargs)
        adults = adults_from_counts(stage_counts)
        records.append({**stage_counts, "generation": offset + 1})

    return records


def _tidy_rows(
    records: list[dict],
    *,
    phase: str,
    replicate: int,
    release_frequency: float,
) -> list[dict]:
    """Melt stage-count dicts into one row per (generation, genotype)."""
    rows = []
    for record in records:
        for genotype in GENOTYPES:
            rows.append(
                {
                    "replicate": replicate,
                    "release_frequency": release_frequency,
                    "generation": record["generation"],
                    "phase": phase,
                    "genotype": genotype,
                    "eggs": record[f"Eggs_{genotype}"],
                    "larvae": record[f"Larvae_{genotype}"],
                    "starved": record[f"Starved_{genotype}"],
                    "timed_out": record[f"TimedOut_{genotype}"],
                    "pupae": record[f"Pupae_{genotype}"],
                    "adults": record[f"Adults_{genotype}"],
                }
            )
    return rows


def _state_rows(
    adults: np.ndarray,
    *,
    generation: int,
    phase: str,
    replicate: int,
    release_frequency: float,
) -> list[dict]:
    """Rows for a state that was assembled rather than simulated.

    The seeded population and the moment of release are both states, not
    cycles: no eggs were laid to reach them, so every stage column but
    ``adults`` is missing rather than zero.  Recording them as NaN keeps a
    survival rate from being computed against an egg cohort that never existed.
    """
    return [
        {
            "replicate": replicate,
            "release_frequency": release_frequency,
            "generation": generation,
            "phase": phase,
            "genotype": genotype,
            "eggs": np.nan,
            "larvae": np.nan,
            "starved": np.nan,
            "timed_out": np.nan,
            "pupae": np.nan,
            "adults": int(adults[index]),
        }
        for index, genotype in enumerate(GENOTYPES)
    ]


def simulate_single_location_releases(
    release_frequencies=(0.1, 0.3, 0.5),
    *,
    conversion_rate: float = 0.0,
    genotype_penalties: dict | None = None,
    param_tag: str | None = None,
    n_replicates: int = 3,
    n_burn_in_generations: int = 5,
    n_generations: int = 6,
    initial_ww: float = 70.0,
    release_mode: str = "additive",
    eggs_per_female: int = 30,
    pond_carrying_capacity: int = POND_CARRYING_CAPACITY,
    temperature: int = 20,
    feeding_type: str = "interference",
    seed: int | None = None,
    n_workers: int | None = None,
) -> pd.DataFrame:
    """Release one gene-drive architecture at several frequencies and record it all.

    See the module docstring for the protocol.  Every simulated generation is
    returned in full, so the abundance, frequency and mortality figures are all
    views of one table rather than three separate runs.

    **Arguments**
    1. release_frequencies: Sequence[float] = (0.1, 0.3, 0.5)
        - GG proportions immediately after release; each must lie in [0, 1)
    2. conversion_rate: float = 0.0
        - the drive's WG -> GG conversion probability
    3. genotype_penalties: dict | None = None
        - the drive's fitness penalties; None uses the IBM's defaults
    4. param_tag: str | None = None
        - provenance marker for a set loaded from a Pareto frontier, recorded
          in the output filenames
    5. n_replicates: int = 3
        - independent burn-in/release replicates; small by default because each
          one costs minutes
    6. n_burn_in_generations: int = 5
        - wild-type generations before the release
    7. n_generations: int = 6
        - generations simulated after the release
    8. initial_ww: float = 70.0
        - wild-type adults the burn-in starts from.  The default is already
          close to the equilibrium this model settles at, so the burn-in has
          little work to do
    9. release_mode: str = "additive"
        - see :func:`apply_release`
    10. eggs_per_female: int = 30
    11. pond_carrying_capacity: int = POND_CARRYING_CAPACITY
    12. temperature: int = 20
    13. feeding_type: str = "interference"
    14. seed: int | None = None
        - master seed; generated and printed if omitted
    15. n_workers: int | None = None
        - worker processes; defaults to :func:`cpu_resources.available_cpu_count`

    **Returns**
    - a tidy DataFrame with one row per (replicate, release frequency,
      generation, genotype) and the columns in :data:`STAGE_COLUMNS`, plus the
      run's metadata as constant columns so the file stands on its own.
    """
    release_frequencies = list(release_frequencies)
    if not release_frequencies:
        raise ValueError("release_frequencies must contain at least one frequency.")
    for frequency in release_frequencies:
        if not 0.0 <= frequency < 1.0:
            raise ValueError(
                f"release_frequency {frequency} is outside [0, 1); a frequency "
                "of 1 would require an unbounded number of released adults."
            )
    if n_replicates < 1:
        raise ValueError(f"n_replicates must be at least 1; got {n_replicates}.")
    if n_burn_in_generations < 1:
        raise ValueError(
            f"n_burn_in_generations must be at least 1; got {n_burn_in_generations}. "
            "The release is into an equilibrated population, which takes a burn-in."
        )
    if n_generations < 1:
        raise ValueError(f"n_generations must be at least 1; got {n_generations}.")

    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1)[0])
        print(f"No seed supplied; using generated seed {seed}")

    if n_workers is None:
        n_workers = available_cpu_count()

    burn_in_seeds, release_seeds = (
        sequence.generate_state(n_replicates)
        for sequence in np.random.SeedSequence(seed).spawn(2)
    )

    generation_kwargs = {
        "conversion_rate": conversion_rate,
        "genotype_penalties": genotype_penalties,
        "eggs_per_female": eggs_per_female,
        "pond_carrying_capacity": pond_carrying_capacity,
        "temperature": temperature,
        "feeding_type": feeding_type,
    }

    total_cycles = n_replicates * (
        n_burn_in_generations + len(release_frequencies) * n_generations
    )
    print(
        f"Simulating {total_cycles} IBM cycles "
        f"({n_replicates} replicates x [{n_burn_in_generations} burn-in + "
        f"{len(release_frequencies)} frequencies x {n_generations} generations]) "
        f"across {n_workers} workers."
    )

    # ── 1. Wild-type burn-in, one per replicate and shared across frequencies ─
    burn_in_kwargs = {
        "initial_ww": initial_ww,
        "n_generations": n_burn_in_generations,
        "eggs_per_female": eggs_per_female,
        "pond_carrying_capacity": pond_carrying_capacity,
        "temperature": temperature,
        "feeding_type": feeding_type,
    }
    with multiprocess.Pool(n_workers) as pool:
        burn_ins = pool.map(
            _burn_in_worker,
            [(int(burn_in_seed), burn_in_kwargs) for burn_in_seed in burn_in_seeds],
        )

    equilibrium_adults = [adults_from_counts(records[-1]) for records in burn_ins]
    summaries = [equilibrium_summary(records) for records in burn_ins]
    pooled_mean = float(np.mean([summary["mean"] for summary in summaries]))
    print(
        f"Burn-in complete after {n_burn_in_generations} generations: "
        f"equilibrium total adults = {pooled_mean:.1f} "
        f"(mean over the last {_EQUILIBRIUM_WINDOW} generations of each replicate)"
    )
    for replicate, summary in enumerate(summaries):
        marker = "" if summary["settled"] else "  <- still trending"
        print(
            f"  replicate {replicate}: mean {summary['mean']:.1f} "
            f"(range {summary['minimum']:.0f}-{summary['maximum']:.0f}), "
            f"relative drift {summary['drift']:.2f}{marker}"
        )
    if not all(summary["settled"] for summary in summaries):
        print(
            "  Some replicates had not settled: raise --n-burn-in-generations, "
            "or start closer to equilibrium with --initial-ww."
        )

    # ── 2. Release, and advance every branch ─────────────────────────────────
    branches = []
    for replicate in range(n_replicates):
        for frequency in release_frequencies:
            released = apply_release(
                equilibrium_adults[replicate], frequency, mode=release_mode
            )
            branches.append((replicate, frequency, released))

    with multiprocess.Pool(n_workers) as pool:
        branch_records = pool.map(
            _post_release_worker,
            [
                (
                    released,
                    int(release_seeds[replicate]),
                    n_generations,
                    generation_kwargs,
                )
                for replicate, _, released in branches
            ],
        )

    # ── 3. Assemble the tidy table ───────────────────────────────────────────
    rows: list[dict] = []
    for (replicate, frequency, released), records in zip(branches, branch_records):
        rows.extend(
            _state_rows(
                np.array([int(initial_ww), 0, 0]),
                generation=-n_burn_in_generations - 1,
                phase=PHASE_SEED,
                replicate=replicate,
                release_frequency=frequency,
            )
        )
        rows.extend(
            _tidy_rows(
                burn_ins[replicate],
                phase=PHASE_BURN_IN,
                replicate=replicate,
                release_frequency=frequency,
            )
        )
        rows.extend(
            _state_rows(
                released,
                generation=0,
                phase=PHASE_RELEASE,
                replicate=replicate,
                release_frequency=frequency,
            )
        )
        rows.extend(
            _tidy_rows(
                records,
                phase=PHASE_POST_RELEASE,
                replicate=replicate,
                release_frequency=frequency,
            )
        )

    dataframe = pd.DataFrame(rows)
    # Metadata as constant columns: a CSV cannot carry attributes, and a figure
    # rebuilt from the file months later still needs to know what was simulated.
    dataframe["conversion_rate"] = conversion_rate
    dataframe["temperature"] = temperature
    dataframe["release_mode"] = release_mode
    dataframe["param_tag"] = param_tag if param_tag else "default"
    dataframe["seed"] = seed
    return dataframe


def add_derived_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add the per-row quantities the figures read, leaving the counts intact.

    Everything here is a function of the recorded counts, so it is derived on
    demand rather than stored: a CSV written by an older run picks up new
    derived columns for free.

    Adds, per row: ``genotype_frequency`` (of adults, within its generation),
    ``egg_to_adult_survival``, ``larva_to_pupa_survival``, and the larval fate
    shares ``pupated_share`` / ``starved_share`` / ``timed_out_share``.  Adds,
    per (replicate, frequency, generation): ``total_adults`` and
    ``allele_frequency``, the frequency of the G allele among adults.
    """
    dataframe = dataframe.copy()
    keys = ["replicate", "release_frequency", "generation"]

    totals = dataframe.groupby(keys)["adults"].transform("sum")
    dataframe["total_adults"] = totals
    dataframe["genotype_frequency"] = np.where(
        totals > 0, dataframe["adults"] / totals, np.nan
    )

    # G alleles: two per GG adult, one per heterozygote, out of two per adult.
    allele_weights = dataframe["genotype"].map({"WW": 0.0, "WG": 1.0, "GG": 2.0})
    g_alleles = (dataframe["adults"] * allele_weights).groupby(
        [dataframe[key] for key in keys]
    ).transform("sum")
    dataframe["allele_frequency"] = np.where(
        totals > 0, g_alleles / (2.0 * totals), np.nan
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        dataframe["egg_to_adult_survival"] = np.where(
            dataframe["eggs"] > 0, dataframe["adults"] / dataframe["eggs"], np.nan
        )
        dataframe["larva_to_pupa_survival"] = np.where(
            dataframe["larvae"] > 0, dataframe["pupae"] / dataframe["larvae"], np.nan
        )
        for fate, column in (
            ("pupated_share", "pupae"),
            ("starved_share", "starved"),
            ("timed_out_share", "timed_out"),
        ):
            dataframe[fate] = np.where(
                dataframe["larvae"] > 0,
                dataframe[column] / dataframe["larvae"],
                np.nan,
            )

    return dataframe


def classify_outcomes(
    dataframe: pd.DataFrame,
    *,
    loss_threshold: float = 0.05,
    fixation_threshold: float = 0.95,
) -> pd.DataFrame:
    """Label how each replicate ended, by release frequency.

    An underdominant drive has a threshold: from the same release frequency,
    some replicates carry the construct to fixation and others lose it, and a
    mean allele frequency halfway between the two describes neither.  The
    outcome shares do.

    **Returns**
    - a DataFrame with one row per (release frequency, replicate) and an
      ``outcome`` column: ``extinct`` (no adults left at all), ``lost``
      (G allele frequency below *loss_threshold*), ``fixed`` (above
      *fixation_threshold*) or ``polymorphic``.
    """
    derived = add_derived_columns(dataframe)
    final = derived[derived["phase"] == PHASE_POST_RELEASE]
    if final.empty:
        return pd.DataFrame(columns=["release_frequency", "replicate", "outcome"])

    last_generation = final["generation"].max()
    final = final[final["generation"] == last_generation]

    summary = (
        final.groupby(["release_frequency", "replicate"])
        .agg(total_adults=("adults", "sum"), allele_frequency=("allele_frequency", "first"))
        .reset_index()
    )

    def _label(row) -> str:
        if row["total_adults"] <= 0:
            return "extinct"
        if row["allele_frequency"] <= loss_threshold:
            return "lost"
        if row["allele_frequency"] >= fixation_threshold:
            return "fixed"
        return "polymorphic"

    summary["outcome"] = summary.apply(_label, axis=1)
    return summary
