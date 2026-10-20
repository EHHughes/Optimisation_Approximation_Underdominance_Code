"""Standalone plotting functions for comparing IBM runs to surrogate model predictions."""

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import multiprocess
import numpy as np
import pandas as pd
import plot_style
import torch
from matplotlib.colors import to_rgb


def _lighten(color, amount: float = 0.55):
    """Blend *color* towards white by *amount* (0 = unchanged, 1 = white)."""
    r, g, b = to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)
from IBM_Implementations.input_output_pairs import (
    breed_adults_to_eggs,
    generate_pairs_from_fixed_adults,
    run_ibm_from_eggs,
)
from paths import IBM_OUTPUT_DIR, IMAGES_DIR

from IBM_Emulation.dataloaders import embed_homozygotes, select_homozygotes
from IBM_Emulation.surrogates import load_pretrained_neural_net
from IBM_Emulation.variational_approximations import load_pretrained_maf


def compare_ibm_to_neural_net(
    adults_WW: int = 200,
    adults_WG: int = 200,
    adults_GG: int = 200,
    n_ibm_runs: int = 10,
    fixed_temperature: int = 20,
    feeding_type: str = "interference",
    seed: int | None = None,
    model_seed: int | None = None,
    homozygotes_only: bool = False,
) -> plt.Figure:
    """Compare stochastic IBM outputs to a neural net prediction for fixed inputs.

    Runs the IBM n_ibm_runs times from the same fixed adult population, then
    overlays the (deterministic) single-step adults→adults NN prediction. This
    lets you see how well the surrogate captures the IBM's central tendency
    relative to its stochastic spread.

    Parameters
    ----------
    adults_WW, adults_WG, adults_GG : int
        Fixed input adult counts per genotype.
    n_ibm_runs : int
        Number of repeated IBM simulations to run.
    fixed_temperature : int
        Temperature for the IBM in °C (must be in [20, 22, 24, 26, 28, 30]).
    feeding_type : str
        "interference" or "exploitative".
    seed : int | None
        Random seed for IBM runs.
    model_seed : int | None
        Seed identifying which trained surrogate to load (matches the seed
        encoded in the saved model's filename). None loads the unseeded model.

    Returns
    -------
    matplotlib.figure.Figure
        The figure, with seed and model_seed recorded in its title for
        reproducibility.
    """
    # --- IBM runs -------------------------------------------------------
    ibm_df = generate_pairs_from_fixed_adults(
        adults_WW=adults_WW,
        adults_WG=adults_WG,
        adults_GG=adults_GG,
        n_trials=n_ibm_runs,
        fixed_temperature=fixed_temperature,
        feeding_type=feeding_type,
        seed=seed,
    )
    ibm_out = {
        "WW": ibm_df["output_adults_WW"].values,
        "WG": ibm_df["output_adults_WG"].values,
        "GG": ibm_df["output_adults_GG"].values,
    }
    ibm_out["Total"] = ibm_out["WW"] + ibm_out["WG"] + ibm_out["GG"]

    # --- Neural net prediction (single-step adults → adults) ------------
    adults_arr = np.array([adults_WW, adults_WG, adults_GG], dtype=np.float32)

    net, scalings = load_pretrained_neural_net(
        with_genetics=True, seed=model_seed, homozygotes_only=homozygotes_only
    )
    x = torch.tensor(adults_arr / scalings[0]).float().reshape(1, 1, 3)
    with torch.no_grad():
        nn_pred = net(x).squeeze().numpy() * scalings[1]

    nn_label = "NN (adults$\\to$adults)"

    nn_pred_by_genotype = {
        "WW": nn_pred[0],
        "WG": nn_pred[1],
        "GG": nn_pred[2],
        "Total": nn_pred[0] + nn_pred[1] + nn_pred[2],
    }

    # --- Plot -----------------------------------------------------------
    genotypes = ["WW", "WG", "GG", "Total"]
    colours = ["#0C7BDC", "#FFC20A", "#DC267F", "#984ea3"]
    rng = np.random.default_rng(seed)

    with plt.rc_context(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 13,
            "axes.titlesize": 13,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 12,
            "figure.titlesize": 14,
        }
    ):
        fig, axes = plt.subplots(1, 4, figsize=(14, 5), sharey=False)

        for ax, gtype, colour in zip(axes, genotypes, colours):
            ibm_vals = ibm_out[gtype]
            nn_val = nn_pred_by_genotype[gtype]

            # Individual IBM runs as jittered dots
            jitter = rng.uniform(-0.15, 0.15, size=len(ibm_vals))
            ax.scatter(
                np.ones(len(ibm_vals)) + jitter,
                ibm_vals,
                c=colour,
                s=55,
                alpha=0.8,
                label="IBM runs",
                zorder=3,
            )
            # IBM mean ± SD
            ax.errorbar(
                1,
                np.mean(ibm_vals),
                yerr=np.std(ibm_vals),
                fmt="D",
                color="k",
                markersize=6,
                capsize=5,
                label="IBM mean $\\pm$ SD",
                zorder=4,
            )
            # NN prediction as a horizontal dashed line
            ax.axhline(
                nn_val,
                color="k",
                linestyle="--",
                linewidth=1.5,
                label=nn_label,
                zorder=2,
            )

            ax.set_title(gtype)
            ax.set_ylabel("Output adults")
            ax.set_xticks([])
            ax.legend()

        plt.tight_layout()
        plt.show()

    return fig


def compare_ibm_to_maf(
    adults_WW: int = 200,
    adults_WG: int = 200,
    adults_GG: int = 200,
    n_ibm_runs: int = 10,
    n_maf_samples: int = 10,
    fixed_temperature: int = 20,
    feeding_type: str = "interference",
    seed: int | None = None,
    model_seed: int | None = None,
    homozygotes_only: bool = False,
) -> plt.Figure:
    """Compare stochastic IBM outputs to samples drawn from the trained MAF.

    The MAF is a conditional normalizing flow — unlike the point-prediction NN,
    it models the full conditional distribution p(outputs | inputs), so each
    call to `flow(context).sample()` gives a different plausible output.  This
    lets you see whether the MAF's sampled spread matches the IBM's stochasticity.

    Parameters
    ----------
    adults_WW, adults_WG, adults_GG : int
        Fixed input adult counts per genotype.
    n_ibm_runs : int
        Number of repeated IBM simulations to run.
    n_maf_samples : int
        Number of samples to draw from the MAF conditional distribution.
    fixed_temperature : int
        Temperature for the IBM in °C (must be in [20, 22, 24, 26, 28, 30]).
    feeding_type : str
        "interference" or "exploitative".
    seed : int | None
        Random seed for IBM runs.
    model_seed : int | None
        Seed identifying which trained MAF to load (matches the seed encoded in
        the saved model's filename). None loads the unseeded model.

    Returns
    -------
    matplotlib.figure.Figure
        The figure, with seed and model_seed recorded in its title for
        reproducibility.
    """
    # --- IBM runs -------------------------------------------------------
    ibm_df = generate_pairs_from_fixed_adults(
        adults_WW=adults_WW,
        adults_WG=adults_WG,
        adults_GG=adults_GG,
        n_trials=n_ibm_runs,
        fixed_temperature=fixed_temperature,
        feeding_type=feeding_type,
        seed=seed,
    )
    ibm_out = {
        "WW": ibm_df["output_adults_WW"].values,
        "WG": ibm_df["output_adults_WG"].values,
        "GG": ibm_df["output_adults_GG"].values,
    }
    ibm_out["Total"] = ibm_out["WW"] + ibm_out["WG"] + ibm_out["GG"]

    # --- MAF samples ----------------------------------------------------
    # load_pretrained_maf builds the flow with the same architecture used during
    # training and returns the matching scalings (adults→adults, with genetics).
    flow, scalings = load_pretrained_maf(
        seed=model_seed, homozygotes_only=homozygotes_only
    )
    flow.eval()

    adults_arr = np.array([adults_WW, adults_WG, adults_GG], dtype=np.float32)
    context = torch.tensor(adults_arr / scalings[0]).float()  # shape (3,)
    if homozygotes_only:
        context = select_homozygotes(context)

    with torch.no_grad():
        maf_samples_norm = flow(context).sample((n_maf_samples,))  # (n_maf_samples, 3)
    if homozygotes_only:
        maf_samples_norm = embed_homozygotes(maf_samples_norm)
    maf_samples = np.clip(maf_samples_norm.numpy() * scalings[1], 0.0, None)

    # approx mean. The zero vector starts in the flow's own latent space, so it
    # is as wide as the flow's features, not always three.
    maf_mean = (
        flow(context)
        .transform.inv(torch.zeros(context.shape[-1]))
        .detach()
        .numpy()
        * scalings[1]
    )
    if homozygotes_only:
        maf_mean = embed_homozygotes(maf_mean)
    maf_mean = np.concat(
        (
            maf_mean.ravel(),
            maf_mean.sum().ravel(),
        )
    )

    maf_out = {
        "WW": maf_samples[:, 0],
        "WG": maf_samples[:, 1],
        "GG": maf_samples[:, 2],
    }
    maf_out["Total"] = maf_out["WW"] + maf_out["WG"] + maf_out["GG"]

    # --- Plot -----------------------------------------------------------
    genotypes = ["WW", "WG", "GG", "Total"]
    ibm_colour = "#e41a1c"
    maf_colour = "#377eb8"
    rng = np.random.default_rng(seed)

    with plt.rc_context(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 13,
            "axes.titlesize": 13,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 12,
            "figure.titlesize": 14,
        }
    ):
        fig, axes = plt.subplots(1, 4, figsize=(14, 5), sharey=False)

        for ax, gtype in zip(axes, genotypes):
            ibm_vals = ibm_out[gtype]
            maf_vals = maf_out[gtype]

            jitter_ibm = rng.uniform(-0.1, 0.1, size=len(ibm_vals))
            jitter_maf = rng.uniform(-0.1, 0.1, size=len(maf_vals))

            # IBM runs
            ax.scatter(
                np.ones(len(ibm_vals)) + jitter_ibm,
                ibm_vals,
                c=ibm_colour,
                s=55,
                alpha=0.8,
                label="IBM",
                zorder=3,
            )
            ax.errorbar(
                1,
                np.mean(ibm_vals),
                yerr=np.std(ibm_vals),
                fmt="D",
                color=ibm_colour,
                markersize=7,
                capsize=5,
                label="IBM mean $\\pm$ SD",
                zorder=4,
            )

            # MAF samples
            ax.scatter(
                np.ones(len(maf_vals)) * 2 + jitter_maf,
                maf_vals,
                c=maf_colour,
                s=55,
                alpha=0.8,
                label="MAF",
                zorder=3,
            )
            ax.errorbar(
                2,
                np.mean(maf_vals),
                yerr=np.std(maf_vals),
                fmt="D",
                color=maf_colour,
                markersize=7,
                capsize=5,
                label="MAF mean $\\pm$ SD",
                zorder=4,
            )
            ax.scatter(
                2,
                maf_mean[genotypes.index(gtype)],
                c="k",
                marker="X",
                s=100,
                label="MAF approx mean",
                zorder=5,
            )

            ax.set_title(gtype)
            ax.set_ylabel("Output adults")
            ax.set_xticks([1, 2])
            ax.set_xticklabels(["IBM", "MAF"])
            ax.legend()

        plt.tight_layout()
        plt.show()

    return fig


def _run_ibm_replicate(args: tuple) -> np.ndarray:
    """Run a single IBM replicate trajectory; intended for multiprocess.Pool.map."""
    (
        init_WW,
        init_WG,
        init_GG,
        n_generations,
        fixed_temperature,
        feeding_type,
        rep_seed,
    ) = args
    if rep_seed is not None:
        np.random.seed(rep_seed)
    traj = np.zeros((n_generations + 1, 3), dtype=float)
    traj[0] = [init_WW, init_WG, init_GG]
    ww, wg, gg = init_WW, init_WG, init_GG
    for g in range(n_generations):
        eggs_WW, eggs_WG, eggs_GG = breed_adults_to_eggs(
            ww, wg, gg, pond_carrying_capacity=2800
        )
        result = run_ibm_from_eggs(
            eggs_WW, eggs_WG, eggs_GG, fixed_temperature, feeding_type
        )
        ww, wg, gg = result if result is not None else (0, 0, 0)
        traj[g + 1] = [ww, wg, gg]
    return traj


def compare_multigeneration_ibm_to_nn(
    n_generations: int = 10,
    init_WW: int = 50,
    init_WG: int = 0,
    init_GG: int = 50,
    n_replicates: int = 5,
    fixed_temperature: int = 20,
    feeding_type: str = "interference",
    seed: int | None = None,
    model_seed: int | None = None,
    homozygotes_only: bool = False,
    save_path: str | None = None,
    for_conference: bool = False,
    model_type: str = "nn",
) -> plt.Figure:
    """Compare multi-generation IBM trajectories to a surrogate model.

    Runs n_replicates independent stochastic IBM trajectories for n_generations,
    chaining breed_adults_to_eggs → run_ibm_from_eggs at each step. Overlays
    a single surrogate trajectory iterated each generation. The surrogate is
    either the single-step adults→adults NN (model_type="nn") or a MAF
    (model_type="maf").

    For the MAF, n_maf_trajectories independent stochastic trajectories are run
    (one sample drawn per generation step, negatives clipped to 0), and their
    mean is plotted as the surrogate trajectory.

    Parameters
    ----------
    n_generations : int
        Number of generations to simulate.
    init_WW, init_WG, init_GG : int
        Initial adult counts at generation 0.
    n_replicates : int
        Number of independent stochastic IBM trajectories to run.
    fixed_temperature : int
        Temperature in °C (must be in [20, 22, 24, 26, 28, 30]).
    feeding_type : str
        "interference" or "exploitative".
    seed : int | None
        Random seed passed to np.random for reproducibility.
    model_seed : int | None
        Seed identifying which trained surrogate (NN or MAF) to load (matches
        the seed encoded in the saved model's filename). None loads the
        unseeded model.
    save_path : str | None
        Where to save the figure. Auto-generated from parameters if None.
    model_type : str
        "nn" to use the single-step adults→adults feedforward NN, or "maf" to
        use the MAF.
    n_maf_trajectories : int
        Number of independent stochastic MAF trajectories to average when
        model_type="maf".

    Returns
    -------
    matplotlib.figure.Figure
        The figure. Both seed and model_seed are encoded in the auto-generated
        save_path (and the title) for reproducibility.
    """
    if model_type not in ("nn", "maf"):
        raise ValueError(f"model_type must be 'nn' or 'maf', got {model_type!r}")

    # --- IBM trajectories (parallelised over replicates) ----------------
    # ibm_traj[replicate, generation, genotype_index]  (0=WW, 1=WG, 2=GG)
    #
    # Every replicate needs its OWN numpy seed. multiprocess forks workers, so
    # they all inherit the parent's global RNG state; without a distinct
    # per-replicate seed each worker would draw the identical sequence and every
    # replicate trajectory would land on top of the others. SeedSequence gives
    # independent, well-separated seeds — random entropy when seed is None,
    # reproducible when a seed is supplied.
    rep_seeds = np.random.SeedSequence(seed).generate_state(n_replicates)
    args_list = [
        (
            init_WW,
            init_WG,
            init_GG,
            n_generations,
            fixed_temperature,
            feeding_type,
            int(rep_seeds[r]),
        )
        for r in range(n_replicates)
    ]
    with multiprocess.Pool() as pool:
        results = pool.map(_run_ibm_replicate, args_list)
    ibm_traj = np.array(results)  # (n_replicates, n_generations+1, 3)

    # --- Surrogate trajectory -------------------------------------------
    surrogate_traj = np.zeros((n_generations + 1, 3), dtype=float)
    surrogate_traj[0] = [init_WW, init_WG, init_GG]
    adults = np.array([init_WW, init_WG, init_GG], dtype=np.float32)

    if model_type == "nn":
        net, scalings = load_pretrained_neural_net(
            with_genetics=True, seed=model_seed, homozygotes_only=homozygotes_only
        )
        for g in range(n_generations):
            x = torch.tensor(adults / scalings[0]).float().reshape(1, 1, 3)
            with torch.no_grad():
                adults = scalings[1] * net(x).detach().numpy()
            surrogate_traj[g + 1] = adults
        surrogate_label = "NN (adults$\\to$adults)"

    else:  # maf
        flow, scalings = load_pretrained_maf(
            seed=model_seed, homozygotes_only=homozygotes_only
        )
        flow.eval()
        maf_trajs = np.zeros((n_replicates, n_generations + 1, 3), dtype=float)
        for traj_idx in range(n_replicates):
            maf_trajs[traj_idx, 0] = [init_WW, init_WG, init_GG]
            traj_adults = np.array([init_WW, init_WG, init_GG], dtype=np.float32)
            for g in range(n_generations):
                context = torch.tensor(traj_adults / scalings[0]).float()
                if homozygotes_only:
                    context = select_homozygotes(context)
                with torch.no_grad():
                    sample = flow(context).sample((1,)).numpy().squeeze()  # (3,)
                if homozygotes_only:
                    sample = embed_homozygotes(sample)
                traj_adults = np.clip(sample * scalings[1], 0.0, None).astype(
                    np.float32
                )
                maf_trajs[traj_idx, g + 1] = traj_adults
        surrogate_traj = maf_trajs.mean(axis=0)  # (n_generations+1, 3)
        surrogate_label = f"MAF mean ($n={n_replicates}$ trajectories)"

    # --- Plot -----------------------------------------------------------
    genotypes = ["WW", "WG", "GG", "Total"]
    colours = [
        plot_style.GENOTYPE_COLORS["WW"],
        plot_style.GENOTYPE_COLORS["WG"],
        plot_style.GENOTYPE_COLORS["GG"],
        plot_style.IBM_PALETTE[1],  # purple for the population total
    ]
    generations = np.arange(n_generations + 1)

    ibm_total = ibm_traj.sum(axis=2)  # (n_replicates, n_generations+1)
    surrogate_total = surrogate_traj.sum(axis=1)  # (n_generations+1,)

    # Conference figures follow the shared house style (usetex Computer Modern);
    # the plain (thesis-inline) path keeps the lighter rc overrides.
    rc = (
        plot_style.HOUSE_RC
        if for_conference
        else {
            "font.size": 13,
            "axes.titlesize": 13,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 12,
            "figure.titlesize": 14,
        }
    )
    with plt.rc_context(rc):
        if for_conference:
            fig, axes_2d = plt.subplots(1, 4, figsize=(16, 5), sharex=True, sharey=True)
            axes = list(axes_2d)
        else:
            fig, axes_2d = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
            axes = list(axes_2d.flat)

        for ax, gtype, colour in zip(axes, genotypes, colours):
            if gtype == "Total":
                ibm_data = ibm_total
                surrogate_data = surrogate_total
            else:
                idx = ["WW", "WG", "GG"].index(gtype)
                ibm_data = ibm_traj[:, :, idx]
                surrogate_data = surrogate_traj[:, idx]

            # Individual replicate trajectories are drawn in a lightened tint of
            # the genotype colour so they read as a faint band behind the bold
            # (full-colour) mean line rather than hiding underneath it.
            light_colour = _lighten(colour)
            for r in range(n_replicates):
                ax.plot(
                    generations,
                    ibm_data[r],
                    color=light_colour,
                    alpha=0.9,
                    linewidth=1.0,
                    label="IBM replicate" if r == 0 else None,
                )
                if model_type == "maf":
                    # Also plot individual MAF trajectories as dashed lines
                    maf_traj_r = (
                        maf_trajs[r, :, idx]
                        if gtype != "Total"
                        else maf_trajs[r].sum(axis=1)
                    )
                    ax.plot(
                        generations,
                        maf_traj_r,
                        color="0.6",
                        alpha=0.9,
                        linewidth=1.0,
                        linestyle="--",
                        label="MAF replicate" if r == 0 else None,
                    )
            ax.plot(
                generations,
                ibm_data.mean(axis=0),
                color=colour,
                linewidth=2.0,
                label="IBM mean",
            )
            ax.plot(
                generations,
                surrogate_data,
                color="k",
                linewidth=2.0,
                linestyle="--",
                label=surrogate_label,
            )

            ax.set_title(gtype)
            # Open the axes: drop the top and right spines (keep left/bottom).
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        if for_conference:
            for ax in axes:
                ax.set_xlabel("Generation")
            axes[0].set_ylabel("Adults")
        else:
            axes[2].set_xlabel("Generation")
            axes[3].set_xlabel("Generation")
            axes[0].set_ylabel("Adults")
            axes[2].set_ylabel("Adults")

        style_handles = [
            mlines.Line2D(
                [], [], color="0.7", alpha=0.9, linewidth=1.0, label="IBM replicate"
            ),
            mlines.Line2D([], [], color="k", linewidth=2.0, label="IBM mean"),
        ]
        if model_type == "maf":
            style_handles.append(
                mlines.Line2D(
                    [], [], color="0.6", alpha=0.9, linewidth=1.0, linestyle="--",
                    label="MAF replicate",
                )
            )
        style_handles.append(
            mlines.Line2D(
                [], [], color="k", linewidth=2.0, linestyle="--", label=surrogate_label,
            )
        )
        fig.legend(
            style_handles,
            [h.get_label() for h in style_handles],
            loc="lower center",
            ncol=len(style_handles),
            frameon=False,
        )
        plt.tight_layout(rect=[0, 0.08, 1, 1])

    name = (
        f"multigen_ibm_vs_{model_type}"
        f"_WW{init_WW}_WG{init_WG}_GG{init_GG}"
        f"_{n_replicates}rep_{n_generations}gen"
        f"_T{fixed_temperature}"
        f"_seed{seed}_modelseed{model_seed}"
    )
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved to {save_path}")
    elif for_conference:
        # Keep the conference set together in CONFERENCE_DIR at house dpi.
        plot_style.save_fig(fig, name)
    else:
        save_path = IMAGES_DIR / f"{name}.png"
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()

    return fig


def plot_emergent_density_dependence(
    data_seed: int = 27,
    max_subsample: int = 6000,
    n_bins: int = 45,
    save: bool = True,
):
    """Figure: the emergent (adults-in -> adults-out) density-dependent map.

    Shows how a whole pond of adults maps to the next generation of adults,
    ignoring genotype.  The stochastic IBM realisations are drawn as a scatter
    of ``(input_adults, output_adults)`` pairs from the aggregated training
    file; their binned mean is overlaid as a smooth curve; and a faint ``y = x``
    replacement line makes the emergent regulation visible — the mean lies above
    replacement at low density (over-compensation) and falls well below it at
    high density (the pond's density-dependent carrying capacity).
    """
    pairs = pd.read_csv(IBM_OUTPUT_DIR / f"input_output_pairs_seed={data_seed}.csv")
    x_in = pairs["input_adults"].to_numpy()
    y_out = pairs["output_adults"].to_numpy()
    top = float(x_in.max())

    # Binned mean of the full dataset (the emergent input->output relation).
    bin_edges = np.linspace(0.0, top, n_bins + 1)
    bin_idx = np.clip(np.digitize(x_in, bin_edges) - 1, 0, n_bins - 1)
    bin_centres, bin_means = [], []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() >= 5:
            bin_centres.append(0.5 * (bin_edges[b] + bin_edges[b + 1]))
            bin_means.append(y_out[mask].mean())
    bin_centres = np.array(bin_centres)
    bin_means = np.array(bin_means)

    # Subsample the (up to 40k) points so the scatter stays legible.
    x_plot, y_plot = x_in, y_out
    if x_in.size > max_subsample:
        rng = np.random.default_rng(0)
        idx = rng.choice(x_in.size, size=max_subsample, replace=False)
        x_plot, y_plot = x_in[idx], y_out[idx]

    with plt.rc_context(plot_style.HOUSE_RC):
        fig, ax = plt.subplots(figsize=(6.5, 5.5))

        ax.scatter(
            x_plot,
            y_plot,
            s=6,
            alpha=0.12,
            color=plot_style.GENOTYPE_COLORS["WW"],
            edgecolors="none",
            label="IBM realisations",
            rasterized=True,
        )
        ax.plot(
            [0, top],
            [0, top],
            linestyle=":",
            color="0.4",
            linewidth=1.3,
            label=r"Replacement ($y = x$)",
        )
        ax.plot(
            bin_centres,
            bin_means,
            color=plot_style.GENOTYPE_COLORS["WW"],
            linewidth=3.5,
            alpha=0.85,
            label="IBM mean",
        )

        ax.set_xlim(0, top)
        ax.set_ylim(0, bin_means.max() * 1.25)
        ax.set_xlabel(r"Adults in")
        ax.set_ylabel(r"Adults out (next generation)")
        ax.set_title(r"Emergent density dependence")
        ax.legend(loc="upper right", framealpha=0.9)
        # Open the axes: drop the top and right spines (keep left/bottom).
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        if save:
            plot_style.save_fig(fig, "emergent_density_dependence")

    return fig

    return fig
