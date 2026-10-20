"""Mating and genetics helpers for the agent-based larval model.

Functions for random mating, Mendelian inheritance, gene-drive conversion,
and updating per-larva genotype assignments.
"""

import numpy as np
import pandas as pd

from IBM_Implementations.helpers.model_config import survival_rates


def convert_heterozygote_eggs(
    eggs_WG: int,
    eggs_GG: int,
    heterozygote_conversion_rate: float,
) -> tuple[int, int]:
    """Apply gene-drive conversion to a cohort of eggs.

    Each WG egg is independently converted to GG with probability
    ``heterozygote_conversion_rate``, so the number converted is a Binomial
    draw.  This is the single implementation of the conversion step: it is used
    both by ``_mate_adults`` (conversion at the moment the eggs are laid) and
    by callers that start from a fixed egg cohort and so have no mating step of
    their own.

    A rate of zero short-circuits without drawing, so a conversion-free run
    consumes no random numbers and stays bitwise identical to one made before
    the drive existed.

    Parameters
    ----------
    eggs_WG : int
        Number of heterozygote eggs before conversion.
    eggs_GG : int
        Number of drive-homozygote eggs before conversion.
    heterozygote_conversion_rate : float
        Probability that any one WG egg is converted to GG.

    Returns
    -------
    tuple[int, int]
        The (WG, GG) egg counts after conversion.  Their sum is unchanged.
    """
    if heterozygote_conversion_rate <= 0:
        return eggs_WG, eggs_GG

    converted = np.random.binomial(eggs_WG, heterozygote_conversion_rate)
    return eggs_WG - converted, eggs_GG + converted


def _mate_adults(
    adults_WW: int,
    adults_WG: int,
    adults_GG: int,
    eggs_per_female: int = 30,
    pond_carrying_capacity: float = np.inf,
    heterozygote_conversion_rate: float = 0.0,
    deterministic_sex_split: bool = True,
) -> tuple[int, int, int]:
    """
    Core Mendelian random-mating logic: mate a population of adults to produce eggs.

    Each female mates with a randomly chosen male; offspring genotypes follow
    Mendelian inheritance. An optional gene-drive conversion step converts a
    fraction of WG eggs to GG.

    Parameters
    ----------
    adults_WW, adults_WG, adults_GG : int
        Adult counts by genotype.
    eggs_per_female : int
        Eggs produced per mating female.
    pond_carrying_capacity : float
        Maximum total eggs; excess are discarded at random.
    heterozygote_conversion_rate : float
        Fraction of WG eggs converted to GG (gene-drive effect).
    deterministic_sex_split : bool
        If True (default), assign exactly floor(n/2) of each genotype as female.
        If False, draw the female count from a binomial(n, 0.5).

    Returns
    -------
    tuple[int, int, int]
        Egg counts (WW, WG, GG).
    """
    if adults_WW + adults_WG + adults_GG == 0:
        return 0, 0, 0

    if deterministic_sex_split:
        females_WW = int(np.floor(adults_WW / 2))
        females_WG = int(np.floor(adults_WG / 2))
        females_GG = int(np.floor(adults_GG / 2))
    else:
        females_WW = np.random.binomial(adults_WW, 0.5)
        females_WG = np.random.binomial(adults_WG, 0.5)
        females_GG = np.random.binomial(adults_GG, 0.5)
    males_WW = adults_WW - females_WW
    males_WG = adults_WG - females_WG
    males_GG = adults_GG - females_GG

    mating_females = ["WW"] * females_WW + ["WG"] * females_WG + ["GG"] * females_GG
    potential_males = (
        int(males_WW) * ["WW"] + int(males_WG) * ["WG"] + int(males_GG) * ["GG"]
    )

    if not mating_females or not potential_males:
        return 0, 0, 0

    mating_males = np.random.choice(
        potential_males, size=len(mating_females), replace=True
    )

    offspring_genotypes = []
    for female_genotype, male_genotype in zip(mating_females, mating_males):
        genes_from_female = np.random.choice(
            list(female_genotype), size=eggs_per_female, replace=True
        )
        genes_from_male = np.random.choice(
            list(male_genotype), size=eggs_per_female, replace=True
        )
        for f_allele, m_allele in zip(genes_from_female, genes_from_male):
            offspring_genotypes.append(
                "".join(sorted([f_allele, m_allele], reverse=True))
            )

    if len(offspring_genotypes) > pond_carrying_capacity:
        np.random.shuffle(offspring_genotypes)
        offspring_genotypes = offspring_genotypes[: int(pond_carrying_capacity)]

    offspring_array = np.array(offspring_genotypes)
    eggs_WW = int(np.sum(offspring_array == "WW"))
    eggs_WG = int(np.sum(offspring_array == "WG"))
    eggs_GG = int(np.sum(offspring_array == "GG"))

    eggs_WG, eggs_GG = convert_heterozygote_eggs(
        eggs_WG, eggs_GG, heterozygote_conversion_rate
    )

    return eggs_WW, eggs_WG, eggs_GG


def get_genotype_distribution_for_stage_pop(
    genotype_stage_distribution: pd.DataFrame,
    cycle: int,
    random_mating: bool = True,
    eggs_per_female: int = 30,
    heterozygote_homozygote_conversion_rate: float = 0,
    pond_carrying_capacity: int = np.inf,
) -> np.ndarray:
    """Computing the new genotypes of larvae based on the genotypes of the adults at the end of the previous cycle. Random mating (and thus genotypes can be turned off for backwards compatibility (e.g. for running tests)). Allows for an optional conversion of some heterozygotes to homozygotes to allow for the presence of a gene drive"""
    if not random_mating:
        return np.zeros((12,))

    if cycle == 0:
        raise ValueError("This function can only be called for cycle number >= 1")

    eggs_WW, eggs_WG, eggs_GG = _mate_adults(
        adults_WW=genotype_stage_distribution.loc[cycle - 1, "Adults_WW"],
        adults_WG=genotype_stage_distribution.loc[cycle - 1, "Adults_WG"],
        adults_GG=genotype_stage_distribution.loc[cycle - 1, "Adults_GG"],
        eggs_per_female=eggs_per_female,
        pond_carrying_capacity=pond_carrying_capacity,
        heterozygote_conversion_rate=heterozygote_homozygote_conversion_rate,
        deterministic_sex_split=True,
    )

    population_structure_at_beginning_of_cycle = np.zeros((12,))
    population_structure_at_beginning_of_cycle[0] = eggs_WW
    population_structure_at_beginning_of_cycle[4] = eggs_WG
    population_structure_at_beginning_of_cycle[8] = eggs_GG

    return population_structure_at_beginning_of_cycle


def update_larvae_genotypes(
    larvae_df: pd.DataFrame,
    stage_pop_including_genotypes: pd.DataFrame,
    cycle_idx: int,
    genotype_penalties: dict,
) -> pd.DataFrame:
    """Update larvae genotypes so that the proportion of different genotypes in
    larvae_df matches the proportion given by the stage structure, then apply
    per-parameter penalties for each genotype.

    genotype_penalties must be a dict of dicts, e.g.::

        {
            "WW": {"a_fac": 1.0, "alpha_fac": 1.0, "beta_fac": 1.0, "g_fac": 1.0},
            "WG": {"a_fac": 1.0, "alpha_fac": 1.0, "beta_fac": 1.0, "g_fac": 0.8},
            "GG": {"a_fac": 1.0, "alpha_fac": 1.0, "beta_fac": 1.0, "g_fac": 0.9},
        }

    Each inner dict maps a larvae_df column name to a multiplicative penalty.
    """
    shuffled_row_indices = np.random.permutation(larvae_df.index)

    wg_count = int(stage_pop_including_genotypes.iloc[cycle_idx, 5])
    gg_count = int(stage_pop_including_genotypes.iloc[cycle_idx, 9])

    larvae_df.loc[shuffled_row_indices[:wg_count], "genotype"] = "WG"
    larvae_df.loc[shuffled_row_indices[wg_count : wg_count + gg_count], "genotype"] = (
        "GG"
    )

    for genotype, penalties in genotype_penalties.items():
        mask = larvae_df["genotype"] == genotype
        if not mask.any():
            continue
        for param_fac, penalty in penalties.items():
            if param_fac in larvae_df.columns:
                larvae_df.loc[mask, param_fac] *= penalty

    return larvae_df


def update_stage_pop_with_genotypes_pupae_and_adults(
    state: dict, cycle_idx: int, temperature: float
) -> pd.DataFrame:
    """Update the stage pop with genotypes"""
    if state["pupated_all"][cycle_idx] is not None:
        number_of_pupated_WW = np.sum(
            state["pupated_all"][cycle_idx].loc[:, "genotype"] == "WW"
        )
        number_of_pupated_WG = np.sum(
            state["pupated_all"][cycle_idx].loc[:, "genotype"] == "WG"
        )
        number_of_pupated_GG = np.sum(
            state["pupated_all"][cycle_idx].loc[:, "genotype"] == "GG"
        )

        state["stage_pop_including_genotypes"].loc[cycle_idx, "Pupae_WW"] = (
            number_of_pupated_WW
        )
        state["stage_pop_including_genotypes"].loc[cycle_idx, "Pupae_WG"] = (
            number_of_pupated_WG
        )
        state["stage_pop_including_genotypes"].loc[cycle_idx, "Pupae_GG"] = (
            number_of_pupated_GG
        )

        state["stage_pop_including_genotypes"].loc[cycle_idx, "Adults_WW"] = int(
            np.floor(number_of_pupated_WW * survival_rates(3, temperature))
        )
        state["stage_pop_including_genotypes"].loc[cycle_idx, "Adults_WG"] = int(
            np.floor(number_of_pupated_WG * survival_rates(3, temperature))
        )
        state["stage_pop_including_genotypes"].loc[cycle_idx, "Adults_GG"] = int(
            np.floor(number_of_pupated_GG * survival_rates(3, temperature))
        )
    else:
        state["stage_pop_including_genotypes"].iloc[cycle_idx, 2:4] = 0
        state["stage_pop_including_genotypes"].iloc[cycle_idx, 6:8] = 0
        state["stage_pop_including_genotypes"].iloc[cycle_idx, 10:12] = 0

    return state["stage_pop_including_genotypes"]
