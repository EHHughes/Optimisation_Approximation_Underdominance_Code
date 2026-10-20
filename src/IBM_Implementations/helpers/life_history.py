"""Per-timestep life-stage transition helpers for agent_based_larval_model.

Discrete life-history events that occur each hour of the simulation:
mortality (removal of starved larvae) and metamorphosis (commitment to and
completion of pupation).
"""

import pandas as pd


def remove_dead_larvae(
    larvae_df: pd.DataFrame,
    state: dict,
    cycle: int,
) -> tuple[pd.DataFrame, int]:
    """Remove larvae with depleted reserves and record them as deceased.

    A larva is considered dead when its reserve R drops below zero.  Dead
    individuals are appended to ``state["deceased_all"][cycle]`` and dropped
    from the active population.

    Parameters
    ----------
    larvae_df:
        Active larvae DataFrame.
    state:
        Simulation state dict (``deceased_all`` list is mutated in-place).
    cycle:
        Zero-based cycle index.

    Returns
    -------
    larvae_df:
        Active larvae DataFrame with dead individuals removed.
    death_count:
        Number of larvae that died this hour.
    """
    death_count = 0
    indices_of_deceased = larvae_df.index[larvae_df["R"] < 0].to_list()
    if len(indices_of_deceased) > 0:
        if state["deceased_all"][cycle] is None:
            state["deceased_all"][cycle] = larvae_df.loc[indices_of_deceased].copy()
        else:
            state["deceased_all"][cycle] = pd.concat(
                [state["deceased_all"][cycle], larvae_df.loc[indices_of_deceased]],
                ignore_index=True,
            )
        larvae_df = larvae_df.drop(index=indices_of_deceased)
        larvae_df.reset_index(drop=True, inplace=True)
        death_count = len(indices_of_deceased)
    return larvae_df, death_count


def commit_to_pupation(
    larvae_df: pd.DataFrame,
    t_global: int,
    Rpup: float,
    pup_delay: float,
) -> tuple[pd.DataFrame, int]:
    """Mark larvae that have reached the pupation-reserve threshold.

    Any larva whose reserve R ≥ Rpup and that has not already been committed
    is assigned a future ``pupate_at`` time of ``t_global + pup_delay``.

    Parameters
    ----------
    larvae_df:
        Active larvae DataFrame.
    t_global:
        Current global time-step.
    Rpup:
        Reserve threshold at which a larva commits to pupation.
    pup_delay:
        Number of time-steps between commitment and actual pupation.

    Returns
    -------
    larvae_df:
        Larvae DataFrame with ``pupate_at`` set for newly committed individuals.
    new_commitments:
        Number of larvae newly committed to pupation this hour.
    """
    new_commitments = 0
    if larvae_df.shape[0] > 0:
        pupated_mask = (larvae_df["pupate_at"].isna()) & (larvae_df["R"] >= Rpup)
        larvae_df.loc[pupated_mask, "pupate_at"] = t_global + pup_delay
        new_commitments = int(pupated_mask.sum())
    return larvae_df, new_commitments


def transfer_pupated(
    larvae_df: pd.DataFrame,
    state: dict,
    cycle: int,
    t_global: int,
    pupae_count: int,
) -> tuple[pd.DataFrame, int, int]:
    """Move larvae whose pupation time has arrived into the pupated list.

    Larvae whose ``pupate_at`` value is ≤ ``t_global`` are appended to
    ``state["pupated_all"][cycle]`` and removed from the active population.

    Parameters
    ----------
    larvae_df:
        Active larvae DataFrame.
    state:
        Simulation state dict (``pupated_all`` list is mutated in-place).
    cycle:
        Zero-based cycle index.
    t_global:
        Current global time-step.
    pupae_count:
        Running total of pupae produced so far this cycle.

    Returns
    -------
    larvae_df:
        Active larvae DataFrame with newly pupated individuals removed.
    pupated_now:
        Number of larvae that completed pupation this hour.
    pupae_count:
        Updated running total of pupae (``pupae_count + pupated_now``).
    """
    pupated_now = 0
    if larvae_df.shape[0] > 0:
        to_pup_mask = (~larvae_df["pupate_at"].isna()) & (
            larvae_df["pupate_at"] <= t_global
        )
        to_pup_idx = larvae_df.index[to_pup_mask].to_list()
        if len(to_pup_idx) > 0:
            pupae_count += len(to_pup_idx)
            if state["pupated_all"][cycle] is None:
                state["pupated_all"][cycle] = larvae_df.loc[to_pup_idx].copy()
            else:
                state["pupated_all"][cycle] = pd.concat(
                    [state["pupated_all"][cycle], larvae_df.loc[to_pup_idx]],
                    ignore_index=True,
                )
            larvae_df = larvae_df.drop(index=to_pup_idx)
            pupated_now = len(to_pup_idx)
    return larvae_df, pupated_now, pupae_count
