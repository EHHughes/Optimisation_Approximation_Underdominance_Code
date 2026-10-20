"""This code is to try out a variational approximation approach to emulating the IBM (specifically Masked-Autoregressive Flow or MAF). The hope is that this will do a better job at captuing the stochastic nature of the surrogate and allow for better approximation of the mean + variance of the distribution of outputs"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import plot_style
import torch
import zuko
from paths import SURROGATE_MODELS_DIR

from IBM_Emulation.dataloaders import (
    HOMOZYGOTE_INDICES,
    embed_homozygotes,
    get_test_train_dataloaders,
    get_test_train_datasets,
    report_heterozygote_residual_for_dataset,
    select_homozygotes,
)


def batch_loop(
    data_loader: torch.utils.data.DataLoader,
    flow: zuko.flows.Flow,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    test: bool = True,
):
    """Do a single epoch of training or (if test=True) show the mean loss over the test dataset."""
    losses = []

    for input, output in data_loader:
        loss = -flow(input).log_prob(output).mean()

        if not test:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            scheduler.step() if scheduler is not None else None

        losses.append(loss.detach())

    losses = torch.stack(losses)

    return losses.mean().item(), losses.std().item()


def maf_model_path(
    seed: int | None,
    param_tag: str | None = None,
    homozygotes_only: bool = False,
) -> Path:
    """Return the path the MAF weights are saved to / loaded from for a given seed.

    ``param_tag`` marks a flow trained on data for a non-default gene-drive
    parameter set, mirroring the ``_params=`` marker
    :func:`paths.input_output_pairs_path` puts on the training data so the model
    can be traced back to the drive it emulates. ``None`` -- the
    default-parameter case -- gives the filename used before custom parameter
    sets existed.

    ``homozygotes_only`` marks a two-genotype ``(WW, GG)`` flow, whose feature
    and context widths differ from the three-genotype one so the two could
    never be loaded into each other.
    """
    params = f"_params={param_tag}" if param_tag else ""
    homozygotes = "_homozygotes=True" if homozygotes_only else ""
    return SURROGATE_MODELS_DIR / f"maf_model_seed={seed}{params}{homozygotes}.pt"


def train_maf(
    hidden_features_width: int = 1024,
    hidden_features_depth: int = 5,
    seed: int | None = None,
    data_seed: int = 27,
    test_data_seed: int | None = None,
    param_tag: str | None = None,
    homozygotes_only: bool = False,
):
    """NSF surrogate on the IBM dataset with genetics, with printing to show the loss is evolving - also configured to work on GPU if available.

    Passing ``seed`` controls the numpy and pytorch RNG (and the train/test
    split) so the training run is reproducible. The seed is encoded in the
    saved model's filename so the model can be reproduced from its name.
    ``data_seed`` selects the training dataset and ``test_data_seed``, when
    given, a held-out test set (see :func:`get_test_train_dataloaders`).
    ``param_tag`` selects data generated for a non-default gene-drive parameter
    set and is carried into the saved model's filename.

    ``homozygotes_only`` fits a genuine two-genotype ``(WW, GG)`` flow to the
    heterozygote-free data, for underdominance architectures whose
    heterozygote is non-viable; :func:`make_maf_sampler` wraps it back up as
    the pipeline's three-genotype map.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    train_loader, test_loader, scalings = get_test_train_dataloaders(
        with_genetics=True,
        device=device,
        batch_size=512,
        seed=seed,
        data_seed=data_seed,
        test_data_seed=test_data_seed,
        param_tag=param_tag,
        homozygotes_only=homozygotes_only,
    )

    if homozygotes_only:
        # Forcing WG to zero is an assertion about the architecture, so measure
        # what it discards before spending hours fitting to it.
        report_heterozygote_residual_for_dataset(
            data_seed,
            param_tag=param_tag,
            label=f"MAF training data (seed={data_seed})",
        )

    # define the MAF model
    n_features = len(HOMOZYGOTE_INDICES) if homozygotes_only else 3
    flow = zuko.flows.NSF(
        features=n_features,
        context=n_features,
        transforms=hidden_features_depth,
        hidden_features=[hidden_features_width] * hidden_features_depth,
    )
    flow = flow.to(device)

    optimizer = torch.optim.Adam(flow.parameters(), lr=0.5*1e-3)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.1)

    for epoch in range(300):
        loss_mean, loss_std = batch_loop(
            train_loader,
            flow,
            optimizer,
            test=False,
        )

        print(f"({epoch})", loss_mean, "±", loss_std)
        print(optimizer.param_groups[0]["lr"])
        if epoch % 10 == 0:
            test_loss, test_std = batch_loop(test_loader, flow, optimizer, test=True)
            print(f"Test loss: {test_loss} ± {test_std}")

    torch.save(
        flow.state_dict(),
        maf_model_path(seed, param_tag=param_tag, homozygotes_only=homozygotes_only),
    )


def load_pretrained_maf(
    hidden_features_width: int = 1024,
    hidden_features_depth: int = 5,
    seed: int | None = None,
    data_seed: int = 27,
    param_tag: str | None = None,
    homozygotes_only: bool = False,
) -> zuko.flows.Flow:
    """Rebuild the NSF flow and load pretrained weights.

    ``param_tag`` selects a flow trained for a non-default gene-drive parameter
    set, and is applied both to the weights filename and to the dataset the
    normalising ``scalings`` are recomputed from.

    ``homozygotes_only`` loads a two-genotype ``(WW, GG)`` flow. Unlike the
    feedforward surrogate, the flow itself is returned **unwrapped**: ``flow(x)``
    yields a distribution whose ``sample``, ``log_prob``, ``transform`` and
    ``transform.inv`` are each used somewhere, and standing in for all of that
    would be fragile. Use :func:`make_maf_sampler` to get the three-genotype
    ``(N, 3) -> (N, 3)`` map that ``SpatialSimulation`` wants, and
    :func:`get_maf_mean_prediction` for the mean.
    """
    n_features = len(HOMOZYGOTE_INDICES) if homozygotes_only else 3
    flow = zuko.flows.NSF(
        features=n_features,
        context=n_features,
        transforms=hidden_features_depth,
        hidden_features=[hidden_features_width] * hidden_features_depth,
    )
    flow.load_state_dict(
        torch.load(
            maf_model_path(seed, param_tag=param_tag, homozygotes_only=homozygotes_only),
            weights_only=True,
            map_location=torch.device("cpu"),
        )
    )

    _, _, scalings = get_test_train_dataloaders(
        with_genetics=True,
        device="cpu",
        data_seed=data_seed,
        param_tag=param_tag,
        homozygotes_only=homozygotes_only,
    )
    return flow, scalings


def make_maf_sampler(
    flow: zuko.flows.Flow, homozygotes_only: bool = False
) -> Callable[[torch.Tensor], torch.Tensor]:
    """One draw per call, as the pipeline's three-genotype map.

    ``SpatialSimulation`` wants a plain ``(N, 3) -> (N, 3)`` callable; a zuko
    flow instead wants ``flow(context).sample(...)``. This is that adapter, and
    it is shared rather than redefined at each call site.

    With ``homozygotes_only`` the flow is a two-genotype one: the WG column is
    dropped from the context and a zero one re-inserted in the draw, so callers
    built on three genotypes are unaffected. ``WG = 0`` is an assertion about
    the architecture, not a measurement -- see
    :func:`IBM_Emulation.dataloaders.report_heterozygote_residual`.
    """

    def maf_sample(x: torch.Tensor) -> torch.Tensor:
        context = select_homozygotes(x) if homozygotes_only else x
        with torch.no_grad():
            sample = flow(context).sample((1,)).squeeze(0)
        return embed_homozygotes(sample) if homozygotes_only else sample

    return maf_sample


def plot_maf_predictions(
    flow: zuko.flows.Flow, data_loader: torch.utils.data.DataLoader
):
    """Plot the predictions of the MAF model against the true outputs.

    ``flow`` is the flow itself, conditioned here on the input genotype counts;
    a zuko flow is not subscriptable, so it must be unpacked from whatever
    ``load_pretrained_maf`` returns before it gets here.
    """
    import matplotlib.pyplot as plt

    all_outputs = []
    all_inputs = []
    for input, output in data_loader:
        all_outputs.append(output)
        all_inputs.append(input)

    all_outputs = torch.cat(all_outputs, dim=0)
    all_inputs = torch.cat(all_inputs, dim=0)

    with torch.no_grad():
        predicted_outputs = flow(all_inputs).sample((10,)).mean(dim=0)

    n_features = all_outputs.shape[1]
    plt.figure(figsize=(4 * n_features, 4))
    for i in range(n_features):
        plt.subplot(1, n_features, i + 1)
        plt.scatter(
            all_outputs[:, i].numpy(), predicted_outputs[:, i].numpy(), alpha=0.5
        )
        plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), "k--")
        plt.xlabel("True Output")
        plt.ylabel("Predicted Output")
        plt.title(f"Output {i + 1}")
    plt.tight_layout()
    plt.show()


def compare_pretrained_maf_to_data(
    seed: int | None = None,
    data_seed: int = 27,
    test_data_seed: int | None = None,
    param_tag: str | None = None,
    homozygotes_only: bool = False,
):
    """Plot a pretrained flow's predictions against the data it was fit to.

    ``param_tag`` selects the non-default gene-drive parameter set the flow was
    trained for, so both the weights and the data are the ones belonging to that
    drive design.
    """
    flow, _scalings = load_pretrained_maf(
        seed=seed,
        data_seed=data_seed,
        param_tag=param_tag,
        homozygotes_only=homozygotes_only,
    )
    train_loader, test_loader, scalings = get_test_train_dataloaders(
        with_genetics=True,
        seed=seed,
        data_seed=data_seed,
        test_data_seed=test_data_seed,
        param_tag=param_tag,
        homozygotes_only=homozygotes_only,
    )
    plot_maf_predictions(flow, test_loader)
    plot_maf_predictions(flow, train_loader)


def get_maf_mean_prediction(
    flow: zuko.flows.Flow,
    context: torch.Tensor,
    scalings: tuple,
    return_total: bool = True,
    homozygotes_only: bool = False,
) -> np.ndarray:
    """Get the mean prediction of the MAF model for a given context.

    The base distribution's mean is pushed through the learned transform, so
    the zero vector it starts from must be as wide as the flow's features. The
    flows here always use ``features == context``, so the width is read off the
    context rather than hardcoded -- it was ``np.zeros(3)``, which a
    two-genotype flow could not accept.

    With ``homozygotes_only`` the context is reduced to ``(WW, GG)`` and the
    result widened back out with a zero WG column, so callers keep the
    three-genotype layout.
    """
    if homozygotes_only:
        context = select_homozygotes(context)
    with torch.no_grad():
        base_mean = torch.zeros(context.shape[-1])
        maf_mean = flow(context).transform.inv(base_mean).numpy() * scalings[1]
    if homozygotes_only:
        maf_mean = embed_homozygotes(maf_mean)
    if return_total:
        maf_mean = np.concat(
            (
                maf_mean.ravel(),
                maf_mean.sum().ravel(),
            )
        )
    return maf_mean


def check_normality_of_test_set(
    test_train_seeds: tuple[int, int],
    model_seed: int | None = None,
    save_path: Path | None = None,
    bins: int = 50,
    param_tag: str | None = None,
    homozygotes_only: bool = False,
):
    """Map the held-out test outputs into the MAF's latent space and check they
    follow the standard-normal base distribution.

    A correctly-fit normalising flow pushes the conditional output distribution
    onto its base ``N(0, I)``. Transforming the *observed* test outputs through
    the learned conditional transform (``flow(context).transform(output)``)
    should therefore yield latent coordinates that look like independent
    standard normals. Systematic deviations -- a shifted mean, the wrong spread,
    residual cross-correlation between coordinates, or heavy tails -- flag where
    the flow mis-fits the data it has never seen.

    We visualise this with a corner plot of the latent coordinates: the diagonal
    panels show each marginal histogram overlaid with the expected standard
    normal, and the lower-triangle panels show the pairwise 2D densities (which
    should be uncorrelated isotropic blobs). The per-marginal statistics (sample
    mean/std and the ``scipy.stats.normaltest`` z-statistic and p-value) are
    printed to the console rather than annotated on the figure.

    Parameters
    ----------
    test_train_seeds : tuple[int, int]
        ``(test_seed, train_seed)`` forwarded to :func:`get_test_train_datasets`
        to locate the held-out test CSV and the training CSV used for scaling.
    model_seed : int | None
        Seed identifying the pretrained MAF weights to load.
    save_path : Path | None
        If given the figure is written here; otherwise it is shown.
    bins : int
        Number of histogram bins for the diagonal/2D panels.
    param_tag : str | None
        Selects the flow trained for a non-default gene-drive parameter set,
        and the data generated for it. ``None`` selects the default-parameter
        flow and data.
    homozygotes_only : bool
        Selects the two-genotype ``(WW, GG)`` flow and the heterozygote-free
        data it was fit to. The corner plot then has two panels per side,
        labelled WW and GG.

    Returns
    -------
    np.ndarray
        The ``(N, 3)`` array of latent coordinates, for further analysis.
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    device = torch.device("cpu")

    # The scalings load_pretrained_maf recomputes are discarded here (the
    # dataset below normalises the data), but the file they come from must
    # exist, so point it at the same train seed as the datasets.
    flow, _ = load_pretrained_maf(
        seed=model_seed,
        data_seed=test_train_seeds[1],
        param_tag=param_tag,
        homozygotes_only=homozygotes_only,
    )
    flow.eval()

    # The training set only supplies the (shared) scalings; we diagnose the
    # held-out test set so the check reflects generalisation, not memorisation.
    _, test_dataset, _ = get_test_train_datasets(
        device, test_train_seeds, param_tag=param_tag, homozygotes_only=homozygotes_only
    )

    # Inputs/outputs are already normalised by the dataset, matching how the
    # flow was trained. The context is the input genotype counts; the data we
    # push through the transform is the observed output genotype counts.
    context = test_dataset.inputs.to(device)
    outputs = test_dataset.outputs.to(device)

    with torch.no_grad():
        latents = flow(context).transform(outputs).cpu().numpy()

    dim = latents.shape[1]
    # Index by genotype, not by position: a two-genotype flow's latents are
    # (WW, GG), and slicing the full list would mislabel the second as WG.
    genotype_positions = (
        list(HOMOZYGOTE_INDICES) if homozygotes_only else list(range(dim))
    )
    labels = [[r"$z_{WW}$", r"$z_{WG}$", r"$z_{GG}$"][i] for i in genotype_positions]
    plain_labels = [["z_WW", "z_WG", "z_GG"][i] for i in genotype_positions]
    grid = np.linspace(-4, 4, 200)

    # IBM blue (the house-style WW colour used in the other surrogate plots),
    # with the matching white -> blue sequential colormap so the 2D density
    # panels read as the same scheme as the diagonal histogram fill.
    ibm_blue = plot_style.GENOTYPE_COLORS["WW"]
    ibm_blue_cmap = plot_style.SEQUENTIAL_CMAP

    # Report the per-marginal normality statistics to the console rather than
    # crowding the figure with them.
    print("MAF latent-space normality (target: standard normal):")
    for i in range(dim):
        statistic, p_value = stats.normaltest(latents[:, i])
        print(
            f"  {plain_labels[i]}: mu={latents[:, i].mean():.3f}, "
            f"sigma={latents[:, i].std():.3f}, "
            f"normaltest z={statistic:.3f}, p={p_value:.2e}"
        )

    # Share x within each column (every panel in column j shows variable j on
    # x), so only the bottom row keeps x tick labels. The diagonal histograms
    # have a density y-axis that differs from the 2D panels, so y is shared
    # separately across the 2D panels of each row below.
    # This function is the only figure in the module and sets no rcParams of
    # its own, so it would otherwise inherit whatever global state an earlier
    # import happened to leave behind.  Pin the house spine convention here.
    with plt.rc_context(plot_style.OPEN_SPINES_RC):
        fig, axes = plt.subplots(
            dim, dim, figsize=(3 * dim, 3 * dim), sharex="col"
        )
        for i in range(1, dim):
            for j in range(1, i):
                axes[i, j].sharey(axes[i, 0])

        for i in range(dim):
            for j in range(dim):
                ax = axes[i, j]

                if j > i:
                    ax.axis("off")
                    continue

                if i == j:
                    ax.hist(
                        latents[:, i],
                        bins=bins,
                        density=True,
                        color=ibm_blue,
                        alpha=0.6,
                    )
                    ax.plot(grid, stats.norm.pdf(grid), "k--", lw=1)
                    ax.set_title(labels[i], fontsize=11)
                    ax.set_xlim(-4, 4)
                    # Density scale carries no shared meaning across panels; drop
                    # its tick labels to declutter.
                    ax.tick_params(labelleft=False)
                else:
                    ax.hist2d(
                        latents[:, j],
                        latents[:, i],
                        bins=bins,
                        range=[[-4, 4], [-4, 4]],
                        cmap=ibm_blue_cmap,
                    )
                    ax.set_xlim(-4, 4)
                    ax.set_ylim(-4, 4)
                    # Only the leftmost 2D panel in each row keeps y tick labels.
                    if j > 0:
                        ax.tick_params(labelleft=False)

                if i == dim - 1:
                    ax.set_xlabel(labels[j])
                if j == 0 and i != 0:
                    ax.set_ylabel(labels[i])

        # Shared labels free up space; tighten the inter-panel padding.
        fig.tight_layout(w_pad=0.3, h_pad=0.3)

    if save_path is not None:
        fig.savefig(save_path, dpi=plot_style.SAVE_DPI, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return latents


if __name__ == "__main__":
    seed = np.random.SeedSequence().generate_state(1)[0].item()
    print(f"Training MAF with seed={seed}")

    train_maf(seed=seed)

    compare_pretrained_maf_to_data(seed=seed)
