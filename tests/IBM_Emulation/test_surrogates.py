"""Tests for IBM_Emulation.surrogates.

Two test suites:
  TestSimpleFeedforwardNN            – architecture and forward-pass properties
  TestTrainAndTestLoops              – train_loop / test_loop behaviour
"""

import pytest
import torch
import torch.utils.data
from IBM_Emulation.surrogates import (
    SimpleFeedforwardNN,
    test_loop,
    train_loop,
)

# Prevent pytest from collecting the imported production functions as tests
test_loop.__test__ = False

# ---------------------------------------------------------------------------
# Teacher-network size used in tests
# ---------------------------------------------------------------------------
# The teacher is deliberately small: large architectures are slow on a CPU-only
# machine, and a narrow hidden layer (6) keeps the teacher's map nonlinear
# enough to be a meaningful fitting target while staying quick to train.
MAX_HIDDEN_LAYERS = 4
MAX_HIDDEN_SIZE = 6
INPUT_SIZE = 3
OUTPUT_SIZE = 3

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_target_net(seed: int = 42) -> SimpleFeedforwardNN:
    """Return the largest teacher network the tests use."""
    torch.manual_seed(seed)
    net = SimpleFeedforwardNN(
        input_size=INPUT_SIZE,
        hidden_size=MAX_HIDDEN_SIZE,
        hidden_layers=MAX_HIDDEN_LAYERS,
        output_size=OUTPUT_SIZE,
    )
    net.apply(init_weights_high_variance)
    return net


def init_weights_high_variance(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.orthogonal_(m.weight, gain=1.5)
        torch.nn.init.uniform_(m.bias, -0.5, 0.5)


def _generate_synthetic_data(
    net: SimpleFeedforwardNN,
    n_samples: int = 1000,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward-pass random inputs through *net* (no grad) to get clean labels."""
    torch.manual_seed(seed)
    inputs = torch.rand(n_samples, INPUT_SIZE)
    net.eval()

    with torch.no_grad():
        outputs = net(inputs)
    return inputs, outputs


def _make_dataloader(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    batch_size: int = 64,
    shuffle: bool = True,
) -> torch.utils.data.DataLoader:
    dataset = torch.utils.data.TensorDataset(inputs, outputs)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _train_net(
    net: SimpleFeedforwardNN,
    train_loader: torch.utils.data.DataLoader,
    n_epochs: int = 40,
    lr: float = 0.001,
    seed: int = 0,
) -> SimpleFeedforwardNN:
    """Train *net* for *n_epochs* and return it."""
    torch.manual_seed(seed)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    for _ in range(n_epochs):
        train_loop(
            train_loader, net, criterion, optimizer, batch_size=train_loader.batch_size
        )
        scheduler.step()
    return net


# ---------------------------------------------------------------------------
# 1. SimpleFeedforwardNN – architecture & forward-pass tests
# ---------------------------------------------------------------------------


class TestSimpleFeedforwardNN:
    """Verify that SimpleFeedforwardNN has the documented architecture and that
    its forward pass behaves correctly."""

    def test_output_shape_matches_output_size(self):
        net = SimpleFeedforwardNN(3, 64, 2, 3)
        x = torch.rand(10, 3)
        assert net(x).shape == (10, 3)

    @pytest.mark.parametrize("n_layers", [1, 2, 3, 4, 5])
    def test_hidden_modulelist_depth(self, n_layers):
        """hidden_layers ModuleList should contain exactly hidden_layers entries."""
        net = SimpleFeedforwardNN(3, 64, n_layers, 3)
        assert len(net.hidden_layers) == n_layers

    @pytest.mark.parametrize("hidden_size", [64, 256, 2048])
    def test_internal_width_is_double_hidden_size(self, hidden_size):
        """hidden2 and every hidden_layers entry should use 2*hidden_size neurons."""
        net = SimpleFeedforwardNN(INPUT_SIZE, hidden_size, 2, OUTPUT_SIZE)
        assert net.hidden2[0].out_features == 2 * hidden_size
        for layer in net.hidden_layers:
            assert layer[0].in_features == 2 * hidden_size
            assert layer[0].out_features == 2 * hidden_size

    def test_output_layer_width(self):
        net = SimpleFeedforwardNN(INPUT_SIZE, 64, 2, OUTPUT_SIZE)
        assert net.out.in_features == 2 * 64
        assert net.out.out_features == OUTPUT_SIZE

    def test_gradients_flow_through_all_named_parameters(self):
        net = SimpleFeedforwardNN(3, 64, 3, 3)
        loss = net(torch.rand(4, 3)).sum()
        loss.backward()
        for name, param in net.named_parameters():
            assert param.grad is not None, f"No gradient for parameter '{name}'"

    def test_batch_independence(self):
        """Output for each sample must be identical whether run solo or in a batch."""
        net = SimpleFeedforwardNN(3, 64, 2, 3)
        net.eval()
        x = torch.rand(6, 3)
        batch_out = net(x)
        solo_out = torch.cat([net(x[i : i + 1]) for i in range(6)])
        assert torch.allclose(batch_out, solo_out, atol=1e-5)

    def test_max_complexity_forward_pass(self):
        """The largest teacher network should produce finite outputs."""
        net = _make_target_net()
        inputs, _ = _generate_synthetic_data(net, n_samples=16)
        out = net(inputs)
        assert out.shape == (16, OUTPUT_SIZE)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


# ---------------------------------------------------------------------------
# 2. train_loop / test_loop
# ---------------------------------------------------------------------------


class TestTrainAndTestLoops:
    """train_loop should reduce loss; test_loop should return a non-negative float."""

    @pytest.fixture(scope="class")
    def random_loaders(self):
        """Small dataloaders with random (non-learnable) targets."""
        torch.manual_seed(0)
        inputs = torch.rand(256, 3)
        outputs = torch.rand(256, 3)
        ds = torch.utils.data.TensorDataset(inputs, outputs)
        loader = torch.utils.data.DataLoader(ds, batch_size=32)
        return loader

    def test_test_loop_returns_float(self, random_loaders):
        net = SimpleFeedforwardNN(3, 64, 1, 3)
        result = test_loop(
            random_loaders, net, torch.nn.MSELoss(), print_test_error=False
        )
        assert isinstance(result, float)

    def test_test_loop_non_negative(self, random_loaders):
        net = SimpleFeedforwardNN(3, 64, 1, 3)
        result = test_loop(
            random_loaders, net, torch.nn.MSELoss(), print_test_error=False
        )
        assert result >= 0.0

    def test_train_loop_reduces_loss(self, random_loaders):
        """Five epochs of Adam on a memorisable dataset must strictly reduce the loss."""
        torch.manual_seed(1)
        net = SimpleFeedforwardNN(3, 256, 2, 3)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
        before = test_loop(random_loaders, net, criterion, print_test_error=False)
        for _ in range(5):
            train_loop(random_loaders, net, criterion, optimizer, batch_size=32)
        after = test_loop(random_loaders, net, criterion, print_test_error=False)
        assert after < before, f"Loss did not decrease: {before:.4f} -> {after:.4f}"

    def test_train_loop_does_not_modify_model_in_eval_mode_labels(self):
        """test_loop must leave the model in eval mode and not accumulate gradients."""
        torch.manual_seed(2)
        net = SimpleFeedforwardNN(3, 64, 1, 3)
        net.eval()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.rand(32, 3), torch.rand(32, 3)),
            batch_size=8,
        )
        test_loop(loader, net, torch.nn.MSELoss(), print_test_error=False)
        # No parameter should have accumulated a gradient
        for param in net.parameters():
            assert param.grad is None
