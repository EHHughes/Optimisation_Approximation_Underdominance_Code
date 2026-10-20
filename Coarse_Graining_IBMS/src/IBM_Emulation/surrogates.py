"""Module to create a simple feedforward neural network for learning the surrogate model for our input-output pair dataset"""

import os
import tempfile
from pathlib import Path

import matplotlib
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import plot_style
import ray
import torch
from paths import IMAGES_DIR, SURROGATE_MODELS_DIR
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from torch import nn

from .dataloaders import (
    HOMOZYGOTE_INDICES,
    embed_homozygotes,
    get_test_train_dataloaders,
    get_test_train_datasets,
    report_heterozygote_residual_for_dataset,
    select_homozygotes,
)

_MOSQUITOS_DIR = str(Path(__file__).resolve().parent.parent.parent)

params = {
    "text.latex.preamble": r"\usepackage{gensymb}",
    "image.origin": "lower",
    "image.interpolation": "nearest",
    "axes.grid": False,
    "savefig.dpi": plot_style.SAVE_DPI,
    "axes.labelsize": 12,  # fontsize for x and y labels (was 10)
    "axes.titlesize": 12,
    "font.size": 10,  # was 10
    "legend.fontsize": 10,  # was 10
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "text.usetex": True,
    "figure.figsize": [6, 4],
    "figure.autolayout": True,
    "font.family": "serif",
    **plot_style.OPEN_SPINES_RC,
}
matplotlib.rcParams.update(params)


def construct_surrogate_function(
    batch_size: int = 256,
    with_genetics: bool = False,
    seed: int | None = None,
    data_seed: int = 27,
    test_data_seed: int | None = None,
    param_tag: str | None = None,
    homozygotes_only: bool = False,
    multi_gpus: bool = False,
) -> None:
    """Construct a surrogate function approximating the IBM using a simple feedforward neural network.

    ``data_seed`` selects the training dataset; ``test_data_seed``, when given,
    selects a held-out test set to evaluate against instead of splitting the
    training data (see :func:`get_test_train_dataloaders`). ``param_tag``
    selects data generated for a non-default gene-drive parameter set and is
    carried into the saved model and figure filenames, so a surrogate for an
    optimised drive is never confused with the default-parameter one.

    ``homozygotes_only`` trains a genuine two-genotype ``(WW, GG) -> (WW, GG)``
    surrogate on the heterozygote-free data, for underdominance architectures
    whose heterozygote is non-viable.  The model is 2-in / 2-out and its
    filename carries a ``_homozygotes=True`` marker;
    :func:`load_pretrained_neural_net` wraps it back up as the pipeline's
    three-genotype map, so nothing downstream changes."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    if homozygotes_only and not with_genetics:
        raise ValueError(
            "homozygotes_only requires with_genetics=True: the aggregated "
            "dataset carries no genotype breakdown to restrict."
        )

    train_dataloader, test_dataloader, scalings = get_test_train_dataloaders(
        batch_size=batch_size,
        with_genetics=with_genetics,
        seed=seed,
        data_seed=data_seed,
        test_data_seed=test_data_seed,
        param_tag=param_tag,
        homozygotes_only=homozygotes_only,
        device=device,
    )

    if homozygotes_only:
        # Forcing WG to zero is an assertion about the architecture, so measure
        # what it discards before spending hours training on it.
        report_heterozygote_residual_for_dataset(
            data_seed,
            param_tag=param_tag,
            label=f"NN training data (seed={data_seed})",
        )

    if with_genetics:
        n_features = len(HOMOZYGOTE_INDICES) if homozygotes_only else 3
        net = SimpleFeedforwardNN(
            input_size=n_features,
            hidden_size=1024,
            hidden_layers=3,
            output_size=n_features,
        )
    else:
        net = SimpleFeedforwardNN(input_size=1, hidden_size=128, output_size=1)
    net = net.to(device)

    if multi_gpus:
        net = nn.DataParallel(net)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.1)

    for epoch in range(900):
        train_loop(train_dataloader, net, criterion, optimizer, batch_size=batch_size)
        scheduler.step()
        current_test_loss = test_loop(
            test_dataloader, net, criterion, print_test_error=False
        )
        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1}\n-------------------------------")
            print(f"Test Loss: {current_test_loss}")

    print("Done!")

    # save model
    torch.save(
        net.state_dict(),
        neural_net_model_path(
            with_genetics, seed, param_tag=param_tag, homozygotes_only=homozygotes_only
        ),
    )

    net = net.cpu()

    _, test_dataloader, scalings = get_test_train_dataloaders(
        batch_size=batch_size,
        with_genetics=with_genetics,
        seed=seed,
        data_seed=data_seed,
        test_data_seed=test_data_seed,
        param_tag=param_tag,
        homozygotes_only=homozygotes_only,
        device=torch.device("cpu"),
    )

    params_marker = f"_params={param_tag}" if param_tag else ""
    homozygotes_marker = "_homozygotes=True" if homozygotes_only else ""
    plot_test_set_predictions(
        # The two-genotype net and its two-column dataset are embedded back into
        # the three-genotype layout, so the figure keeps its WW/WG/GG/Total
        # panels and the WG panel reads as the flat zero the model asserts.
        HomozygoteSurrogate(net) if homozygotes_only else net,
        test_dataloader,
        scalings,
        title="Test set predictions (final)",
        save_path=IMAGES_DIR
        / (
            "Input_Output_vs_NN_Comparison_with_genetics="
            f"{with_genetics}{params_marker}{homozygotes_marker}.png"
        ),
        log_y=True,
        embed=embed_homozygotes if homozygotes_only else None,
    )


def tune_hyperparameters(
    test_train_seeds: tuple[int, int],
    max_num_epochs=900,
    grace_period: int = 300,
    num_trials: int = 3,
    homozygotes_only: bool = False,
):
    """Tune the nn using raytune and train_with_tunable_hyperparameters

    ``homozygotes_only`` searches over two-genotype ``(WW, GG)`` surrogates
    trained on the heterozygote-free data, matching
    :func:`construct_surrogate_function`."""
    if not ray.is_initialized():
        ray.init(
            runtime_env={
                "working_dir": _MOSQUITOS_DIR,
                "excludes": "*.pt",
                "env_vars": {"DEBUG": "1"},
            },
        )
    param_space_and_config = get_param_space_and_config(
        test_train_seeds, homozygotes_only=homozygotes_only
    )
    param_space_and_config["max_num_epochs"] = max_num_epochs

    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=grace_period,
        reduction_factor=2,
    )

    tune_config = tune.TuneConfig(
        metric="loss",
        mode="min",
        scheduler=scheduler,
        num_samples=num_trials,
    )

    trainable = tune.with_resources(
        train_with_tunable_hyperparameters,
        resources={"cpu": os.cpu_count(), "gpu": torch.cuda.device_count()},
    )

    tuner = tune.Tuner(
        trainable, tune_config=tune_config, param_space=param_space_and_config
    )

    results = tuner.fit()

    display_output_and_save_best_model(results)


def get_param_space_and_config(
    test_train_seeds: tuple[int, int], homozygotes_only: bool = False
) -> dict:
    """Returns the default configuration and parameter space for raytune

    ``homozygotes_only`` is carried in the config rather than searched over: it
    fixes the network's input and output width at two, so the trials are all
    two-genotype surrogates.
    """
    param_space_and_config = {
        "hidden_layers": tune.choice(range(1, 6)),
        "hidden_size": tune.choice([2**i for i in range(6, 12)]),
        "batch_size": tune.choice([4, 8, 16, 32, 64, 128]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "decay_rate": tune.loguniform(1e-1, 1e0),
        "step_size": tune.choice([100, 200, 300, 400, 500, 900]),
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "test_train_seeds": test_train_seeds,
        "seed": tune.randint(0, int(1e6)),
        "dataloader_function": load_and_split_dataset,
        "homozygotes_only": homozygotes_only,
    }
    return param_space_and_config


def train_with_tunable_hyperparameters(config: dict):
    """function for training the neural network, set up so we can use raytune to optimise hyperparameters. Takes a config directory with
    {
    hidden_layers: int,
    hidden_size: int,
    batch_size: int,
    learning_rate: float,
    decay_rate: float,
    step_size: int,
    device: torch.device
    test_train_seeds: tuple[int, int],
    seed: int,
    dataloader_function: function,
    homozygotes_only: bool,
    }"""

    rng = torch.Generator().manual_seed(config["seed"])

    n_features = _config_n_features(config)
    net = SimpleFeedforwardNN(
        input_size=n_features,
        hidden_size=config["hidden_size"],
        hidden_layers=config["hidden_layers"],
        output_size=n_features,
    )
    net = net.to(config["device"])
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config["step_size"], gamma=config["decay_rate"]
    )

    checkpoint = tune.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir) / "checkpoint.pt"
            checkpoint_state = torch.load(checkpoint_path, weights_only=True)
            start_epoch = checkpoint_state["epoch"]
            net.load_state_dict(checkpoint_state["net_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    trainloader, valloader = config["dataloader_function"](config=config, rng=rng)

    for epoch in range(start_epoch, config["max_num_epochs"]):
        train_for_one_epoch(
            scheduler,
            trainloader,
            net,
            criterion,
            optimizer,
            device=config["device"],
        )

        avg_val_loss = compute_val_loss(net, valloader, criterion, config["device"])

        if epoch % 10 == 0:
            checkpoint_and_report(net, optimizer, epoch, avg_val_loss)

    print("Finished Training")


def _config_n_features(config: dict) -> int:
    """Genotype width the net should have for this raytune config.

    ``homozygotes_only`` is absent from configs written before the
    heterozygote-free mode existed, so default it to the three-genotype case.
    """
    return len(HOMOZYGOTE_INDICES) if config.get("homozygotes_only") else 3


def load_and_split_dataset(
    config: dict, rng: torch.Generator
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Load the training dataset and split it into train and validation datasets for use in raytune."""
    trainset, _testset, _scalings = get_test_train_datasets(
        device=config["device"],
        test_train_seeds=config["test_train_seeds"],
        homozygotes_only=bool(config.get("homozygotes_only")),
    )

    test_abs = int(len(trainset) * 0.9)
    train_subset, val_subset = torch.utils.data.random_split(
        trainset, [test_abs, len(trainset) - test_abs], generator=rng
    )

    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=min(os.cpu_count(), 2),
    )
    valloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=min(os.cpu_count(), 2),
    )

    return trainloader, valloader


def train_for_one_epoch(
    scheduler: torch.optim.lr_scheduler.StepLR,
    trainloader: torch.utils.data.DataLoader,
    net: torch.nn.Module,
    criterion: torch.nn.MSELoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    """Train the model for one epoch"""
    for batch_idx, data in enumerate(trainloader):
        inputs, outputs = data
        inputs = inputs.to(device)
        outputs = outputs.to(device)

        optimizer.zero_grad()

        pred_outputs = net(inputs)
        loss = criterion(pred_outputs, outputs)
        loss.backward()
        optimizer.step()

    scheduler.step()


def compute_val_loss(
    net: torch.nn.Module,
    valloader: torch.utils.data.DataLoader,
    criterion: torch.nn.MSELoss,
    device: torch.device,
) -> float:
    """Compute the validation loss for the current model"""
    val_loss = 0.0
    val_steps = 0
    for steps, data in enumerate(valloader):
        with torch.no_grad():
            inputs, outputs = data
            inputs = inputs.to(device)
            outputs = outputs.to(device)

            pred_outputs = net(inputs)

            loss = criterion(pred_outputs, outputs)
            val_loss += loss.cpu().numpy()
            val_steps = steps + 1

    avg_loss = val_loss / val_steps
    return avg_loss


def checkpoint_and_report(
    net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    avg_val_loss: float,
):
    """Checkpoint the model and report the validation loss to raytune."""
    checkpoint_data = {
        "epoch": epoch,
        "net_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    with tempfile.TemporaryDirectory() as checkpoint_dir:
        checkpoint_path = Path(checkpoint_dir) / "checkpoint.pt"
        torch.save(checkpoint_data, checkpoint_path)

        checkpoint = tune.Checkpoint.from_directory(checkpoint_dir)
        tune.report(
            {"loss": avg_val_loss},
            checkpoint=checkpoint,
        )


def display_output_and_save_best_model(results):
    """After raytune has finished, get the best result and save the model with the best hyperparameters. Also print the best hyperparameters and the corresponding validation loss and accuracy."""
    best_result = results.get_best_result("loss", "min")
    print(f"Best trial config: {best_result.config}")
    print(f"Best trial final validation loss: {best_result.metrics['loss']}")

    n_features = _config_n_features(best_result.config)
    best_trained_model = SimpleFeedforwardNN(
        input_size=n_features,
        hidden_size=best_result.config["hidden_size"],
        hidden_layers=best_result.config["hidden_layers"],
        output_size=n_features,
    )

    homozygotes = (
        "_homozygotes=True" if best_result.config.get("homozygotes_only") else ""
    )
    torch.save(
        best_trained_model.state_dict(),
        SURROGATE_MODELS_DIR
        / f"surrogate_model_genetics=True_optimised_hyper_parameters{homozygotes}.pt",
    )


class SimpleFeedforwardNN(nn.Module):
    """A simple feedforward neural network for regression tasks."""

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        hidden_layers: int = 4,
        output_size: int = 1,
    ) -> None:
        super().__init__()
        self.hidden1 = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU())
        self.hidden2 = nn.Sequential(nn.Linear(hidden_size, 2 * hidden_size), nn.ReLU())
        self.hidden_layers = nn.ModuleList()
        for _ in range(hidden_layers):
            self.hidden_layers.append(
                nn.Sequential(nn.Linear(2 * hidden_size, 2 * hidden_size), nn.ReLU())
            )
        self.out = nn.Linear(2 * hidden_size, output_size)

    def forward(self, x) -> torch.Tensor:
        """Forward pass through the network."""
        x = self.hidden1(x)
        x = self.hidden2(x)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        x = self.out(x)
        return x


class HomozygoteSurrogate(nn.Module):
    """Present a two-genotype (WW, GG) surrogate as the pipeline's three-genotype map.

    The net inside is genuinely 2-in / 2-out.  This drops the WG column on the
    way in and re-inserts a zero one on the way out, so every caller built on
    the ``(3,) -> (3,)`` / ``(N, 3) -> (N, 3)`` contract -- ``SpatialSimulation``
    and its callers, the wavespeed analysis, the release experiments, every
    plotting function -- works unchanged.

    Being an ``nn.Module`` keeps ``.eval()``, ``.cpu()``, ``.parameters()`` and
    plain ``net(x)`` call syntax working wherever a three-genotype net was used
    before.

    ``WG = 0`` here is an assertion about the architecture, not a measurement of
    the IBM: every WW x GG cross still lays heterozygote eggs, and they reach
    adulthood unless the drive's ``wg_*`` penalties starve them.  Use
    :func:`IBM_Emulation.dataloaders.report_heterozygote_residual` to find out
    how much output mass the assertion discards.
    """

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, x) -> torch.Tensor:
        """Map ``(..., 3)`` genotype abundances to ``(..., 3)``, with WG at zero."""
        return embed_homozygotes(self.net(select_homozygotes(x)))


def test_accuracy(
    net,
    test_train_seeds: tuple[int, int],
    loss_fn,
    device: torch.device = torch.device("cpu"),
    homozygotes_only: bool = False,
):
    _trainset, testset, _scalings = get_test_train_datasets(
        device, test_train_seeds, homozygotes_only=homozygotes_only
    )

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2
    )

    running_loss = 0.0
    size = 0
    with torch.no_grad():
        for data in testloader:
            inputs, outputs = data
            pred_outputs = net(inputs)
            loss = loss_fn(pred_outputs, outputs)

            size += inputs.shape[0]
            running_loss += loss
    return running_loss / size


def train_loop(
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    loss_fn,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    print_train_loss: bool = False,
) -> None:
    """Training loop for on epoch, handling batching etc"""
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            if print_train_loss:
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    loss_fn,
    print_test_error: bool = True,
) -> float:
    """Testing loop to evaluate model performance after each epoch."""
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0.0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    if print_test_error:
        print(f"Test Error: Avg loss: {test_loss:>8f} \n")
    return test_loss


def plot_test_set_predictions(
    net: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    scalings: np.ndarray,
    title: str,
    save_path: str | None = None,
    log_y: bool = False,
    conference: bool = False,
    subsample: int = 1,
    alpha: float = 0.5,
    embed=None,
) -> None:
    """Plot actual vs predicted outputs for a genetics-aware dataset.

    Produces four panels: one per genotype (WW, WG, GG) and a Total panel
    that sums across genotypes. The Total panel includes a dotted x=y identity
    line drawn in the underlying (linear) data space. When log_y=True, all
    y-axes use log scale; otherwise (default) they are linear and truncated at
    y=0. All axes are shared across panels.

    ``subsample`` plots only every ``subsample``-th point to declutter dense
    test sets; ``alpha`` sets the marker opacity.

    ``embed`` widens the dataset's inputs and outputs before the forward pass.
    A two-genotype dataset has no WG column, so pass
    :func:`IBM_Emulation.dataloaders.embed_homozygotes` to restore the
    three-genotype layout that both the four panels and the wrapped surrogate
    expect; the WG panel then shows the flat zero the model asserts.

    conference=True  → 1×4 wide layout suited to slides.
    conference=False → 2×2 layout suited to paper figures (default).
    """
    net.eval()
    inputs = dataloader.dataset.inputs
    outputs = dataloader.dataset.outputs

    genotype_labels = ["WW", "WG", "GG"]
    colours = ["#0C7BDC", "#FFC20A", "#DC267F", "#984ea3"]

    if embed is not None:
        # Widen *before* the forward pass: a two-genotype dataset has no WG
        # column, but the net handed in is the wrapped three-genotype map, so
        # it would reject the narrow tensor.
        inputs = embed(inputs)
        outputs = embed(outputs)

    predicted_y = net(inputs)
    predicted_y = predicted_y.detach().numpy()

    inputs_np = (
        inputs.detach().numpy()
        if isinstance(inputs, torch.Tensor)
        else np.asarray(inputs)
    )
    outputs_np = (
        outputs.detach().numpy()
        if isinstance(outputs, torch.Tensor)
        else np.asarray(outputs)
    )

    # Thin out dense test sets so the scatter is legible.
    if subsample > 1:
        inputs_np = inputs_np[::subsample]
        outputs_np = outputs_np[::subsample]
        predicted_y = predicted_y[::subsample]

    with plt.rc_context(
        {
            "font.size": 13,
            "axes.titlesize": 13,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 12,
            "figure.titlesize": 14,
            # Stated here as well as in the module-level rcParams so the figure
            # gets the house spine convention regardless of import order.
            **plot_style.OPEN_SPINES_RC,
        }
    ):
        if conference:
            fig, axes_2d = plt.subplots(1, 4, figsize=(20, 4), sharex=True, sharey=True)
            axes = axes_2d  # flat list of 4
        else:
            fig, axes_2d = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
            axes = axes_2d.flat  # iterable of 4 in row-major order

        axes = list(axes)

        for i, (label, colour) in enumerate(zip(genotype_labels, colours)):
            jitter = np.random.normal(0, 0.05, size=inputs_np[:, i].shape)
            axes[i].scatter(
                inputs_np[:, i] * scalings[0] + jitter,
                outputs_np[:, i] * scalings[1],
                s=15,
                c=colour,
                alpha=alpha,
            )
            axes[i].scatter(
                inputs_np[:, i] * scalings[0] + jitter,
                predicted_y[:, i] * scalings[1],
                s=15,
                marker="*",
                c=colour,
                alpha=alpha,
            )
            axes[i].set_title(label)

        # Total panel (index 3)
        total_colour = colours[3]
        total_in = inputs_np.sum(axis=1) * scalings[0]
        total_out_actual = outputs_np.sum(axis=1) * scalings[1]
        total_out_pred = predicted_y.sum(axis=1) * scalings[1]

        axes[3].scatter(total_in, total_out_actual, s=15, c=total_colour, alpha=alpha)
        axes[3].scatter(
            total_in, total_out_pred, s=15, marker="*", c=total_colour, alpha=alpha
        )

        # Identity line y=x in the underlying linear space; linspace gives a
        # smooth curve when the y-axis is log-scaled.
        lo, hi = total_in.min(), total_in.max()
        x_line = np.linspace(lo, hi, 300)
        axes[3].plot(x_line, x_line, "k:", linewidth=1.5, zorder=2)

        axes[3].set_title("Total")

        for ax in axes:
            if log_y:
                ax.set_yscale("log")
                ax.set_ylim(bottom=0.5)
            else:
                ax.set_ylim(bottom=0)

        # Axis labels — placement differs by layout
        if conference:
            axes[1].set_xlabel("Input Adults")
            axes[3].set_xlabel("Input Adults")
            axes[0].set_ylabel("Output Adults")
        else:
            axes[2].set_xlabel("Input Adults")
            axes[3].set_xlabel("Input Adults")
            axes[0].set_ylabel("Output Adults")
            axes[2].set_ylabel("Output Adults")

        style_handles = [
            mlines.Line2D(
                [],
                [],
                color="k",
                marker="o",
                linestyle="None",
                markersize=5,
                label="Actual",
            ),
            mlines.Line2D(
                [],
                [],
                color="k",
                marker="*",
                linestyle="None",
                markersize=8,
                label="Predicted",
            ),
            mlines.Line2D(
                [], [], color="k", linestyle=":", linewidth=1.5, label="$x=y$"
            ),
        ]
        fig.legend(
            style_handles,
            [h.get_label() for h in style_handles],
            loc="lower center",
            ncol=len(style_handles),
            frameon=False,
        )
        plt.tight_layout(rect=[0, 0.08, 1, 1])

        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight")
        plt.show()
        plt.close(fig)


def neural_net_model_path(
    with_genetics: bool,
    seed: int | None = None,
    param_tag: str | None = None,
    homozygotes_only: bool = False,
) -> Path:
    """Return the path a feedforward surrogate is saved to / loaded from.

    When ``seed`` is provided, the seed-suffixed filename written by
    ``construct_surrogate_function`` is used so the model can be identified by
    the seed that produced it.

    ``param_tag`` marks a surrogate trained on data for a non-default
    gene-drive parameter set. It mirrors the ``_params=`` marker
    :func:`paths.input_output_pairs_path` puts on the training data, so a model
    can always be traced back to the drive design it emulates. ``None`` -- the
    default-parameter case -- gives exactly the filename used before custom
    parameter sets existed.

    ``homozygotes_only`` marks a two-genotype ``(WW, GG)`` surrogate, which has
    a different input and output width from the three-genotype one and so could
    not be loaded into it. ``False`` -- the default -- gives the filename used
    before the heterozygote-free mode existed.
    """
    params = f"_params={param_tag}" if param_tag else ""
    homozygotes = "_homozygotes=True" if homozygotes_only else ""
    if seed is None:
        return (
            SURROGATE_MODELS_DIR
            / f"surrogate_model_genetics={with_genetics}{params}{homozygotes}.pt"
        )
    return (
        SURROGATE_MODELS_DIR
        / f"surrogate_model_genetics={with_genetics}_seed={seed}{params}{homozygotes}.pt"
    )


def load_pretrained_neural_net(
    with_genetics: bool = False,
    seed: int | None = None,
    data_seed: int = 27,
    param_tag: str | None = None,
    homozygotes_only: bool = False,
) -> tuple[nn.Module, np.ndarray]:
    """Load a pretrained version of the neural net, so that we can run model evaluations on separate devices to where I've done the training.

    Passing ``seed`` selects the seed-suffixed model written by
    ``construct_surrogate_function``; omitting it loads the unseeded filename.
    ``data_seed`` selects the dataset used to recompute the normalising
    ``scalings`` and must match the data the model was trained on.
    ``param_tag`` selects a surrogate trained for a non-default gene-drive
    parameter set, and is applied to both the model filename and the dataset
    the scalings are recomputed from.

    ``homozygotes_only`` loads a two-genotype ``(WW, GG)`` surrogate and returns
    it wrapped in :class:`HomozygoteSurrogate`, so the caller still receives the
    pipeline's three-genotype ``(N, 3) -> (N, 3)`` map and needs no changes of
    its own. It must match the flag the model was trained under: it selects
    both the weights file and the dataset the scalings come from, and the
    scalings differ because they are maxima over the retained columns only.
    """
    net: SimpleFeedforwardNN | HomozygoteSurrogate
    if with_genetics:
        n_features = len(HOMOZYGOTE_INDICES) if homozygotes_only else 3
        net = SimpleFeedforwardNN(n_features, 1024, 3, n_features)
    elif homozygotes_only:
        raise ValueError(
            "homozygotes_only requires with_genetics=True: the aggregated "
            "surrogate carries no genotype breakdown to restrict."
        )
    else:
        net = SimpleFeedforwardNN(1, 128, 1)

    net.load_state_dict(
        torch.load(
            neural_net_model_path(
                with_genetics,
                seed,
                param_tag=param_tag,
                homozygotes_only=homozygotes_only,
            ),
            weights_only=True,
            map_location=torch.device("cpu"),
        )
    )
    net.eval()  # not necessary here but necessary if you have batch_dropout or normalisation layers

    if homozygotes_only:
        net = HomozygoteSurrogate(net)
        net.eval()

    # get scalings for input and output so that we can unscale the outputs of the model when we use it as a surrogate function
    _, _, scalings = get_test_train_dataloaders(
        with_genetics=with_genetics,
        data_seed=data_seed,
        param_tag=param_tag,
        homozygotes_only=homozygotes_only,
    )

    return net, scalings
