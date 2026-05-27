"""Tests to verify that all required metrics are tracked throughout the experiment pipeline.

Covers:
- train_one_epoch returns the required keys
- evaluate returns the required keys
- Full train loop produces per-epoch history with all required keys
- divergence_rate is computed correctly
- aggregate_seeds produces per-key mean/sem for all expected keys
- run_single result dict contains all keys needed for the final report
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytest

from src.training.trainer import train_one_epoch, evaluate, train
from src.analysis.metrics import divergence_rate, aggregate_seeds

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TRAIN_EPOCH_KEYS = {"train_loss", "train_acc", "elapsed_sec"}
EVAL_KEYS = {"test_loss", "test_acc"}
HISTORY_ROW_KEYS = TRAIN_EPOCH_KEYS | EVAL_KEYS | {"epoch"}

# Keys expected in the final result dict produced by run_baseline.run_single
RUN_SINGLE_KEYS = {
    "train_loss", "train_acc", "test_loss", "test_acc",
    "elapsed_sec", "epoch",
    "divergence_rate", "seed", "optimizer", "rho", "run_id", "checkpoint", "history",
}

# Keys expected in each row of the persisted history list
HISTORY_PERSISTENCE_KEYS = {"epoch", "train_loss", "train_acc", "test_loss", "test_acc", "elapsed_sec"}


@pytest.fixture
def tiny_loader():
    """A minimal DataLoader (8 samples, 2 classes) for fast unit tests."""
    torch.manual_seed(0)
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 2, (8,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=4)


@pytest.fixture
def tiny_model():
    """Minimal conv model with 2 output classes."""
    return nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4, 2),
    )


@pytest.fixture
def device():
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# train_one_epoch
# ---------------------------------------------------------------------------

class TestTrainOneEpochMetrics:
    def test_returns_all_required_keys(self, tiny_model, tiny_loader, device):
        optimizer = torch.optim.SGD(tiny_model.parameters(), lr=0.01)
        result = train_one_epoch(tiny_model, optimizer, tiny_loader, nn.CrossEntropyLoss(), device)
        assert TRAIN_EPOCH_KEYS == set(result.keys()), (
            f"Missing keys: {TRAIN_EPOCH_KEYS - set(result.keys())}, "
            f"Extra keys: {set(result.keys()) - TRAIN_EPOCH_KEYS}"
        )

    def test_train_loss_is_positive(self, tiny_model, tiny_loader, device):
        optimizer = torch.optim.SGD(tiny_model.parameters(), lr=0.01)
        result = train_one_epoch(tiny_model, optimizer, tiny_loader, nn.CrossEntropyLoss(), device)
        assert result["train_loss"] > 0.0

    def test_train_acc_in_range(self, tiny_model, tiny_loader, device):
        optimizer = torch.optim.SGD(tiny_model.parameters(), lr=0.01)
        result = train_one_epoch(tiny_model, optimizer, tiny_loader, nn.CrossEntropyLoss(), device)
        assert 0.0 <= result["train_acc"] <= 1.0

    def test_elapsed_sec_is_positive(self, tiny_model, tiny_loader, device):
        optimizer = torch.optim.SGD(tiny_model.parameters(), lr=0.01)
        result = train_one_epoch(tiny_model, optimizer, tiny_loader, nn.CrossEntropyLoss(), device)
        assert result["elapsed_sec"] > 0.0


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

class TestEvaluateMetrics:
    def test_returns_all_required_keys(self, tiny_model, tiny_loader, device):
        result = evaluate(tiny_model, tiny_loader, nn.CrossEntropyLoss(), device)
        assert EVAL_KEYS == set(result.keys()), (
            f"Missing keys: {EVAL_KEYS - set(result.keys())}, "
            f"Extra keys: {set(result.keys()) - EVAL_KEYS}"
        )

    def test_test_loss_is_positive(self, tiny_model, tiny_loader, device):
        result = evaluate(tiny_model, tiny_loader, nn.CrossEntropyLoss(), device)
        assert result["test_loss"] > 0.0

    def test_test_acc_in_range(self, tiny_model, tiny_loader, device):
        result = evaluate(tiny_model, tiny_loader, nn.CrossEntropyLoss(), device)
        assert 0.0 <= result["test_acc"] <= 1.0


# ---------------------------------------------------------------------------
# Full train loop — history shape and keys
# ---------------------------------------------------------------------------

class TestTrainHistoryMetrics:
    def test_history_length_matches_epochs(self, tiny_model, tiny_loader, device):
        optimizer = torch.optim.SGD(tiny_model.parameters(), lr=0.01)
        n_epochs = 3
        history = train(
            tiny_model, optimizer, tiny_loader, tiny_loader,
            nn.CrossEntropyLoss(), device, epochs=n_epochs, verbose=False, compile_model=False,
        )
        assert len(history) == n_epochs

    def test_each_row_has_all_required_keys(self, tiny_model, tiny_loader, device):
        optimizer = torch.optim.SGD(tiny_model.parameters(), lr=0.01)
        history = train(
            tiny_model, optimizer, tiny_loader, tiny_loader,
            nn.CrossEntropyLoss(), device, epochs=2, verbose=False, compile_model=False,
        )
        for i, row in enumerate(history):
            missing = HISTORY_ROW_KEYS - set(row.keys())
            assert not missing, f"Row {i} missing keys: {missing}"

    def test_epoch_numbers_are_sequential(self, tiny_model, tiny_loader, device):
        optimizer = torch.optim.SGD(tiny_model.parameters(), lr=0.01)
        history = train(
            tiny_model, optimizer, tiny_loader, tiny_loader,
            nn.CrossEntropyLoss(), device, epochs=3, verbose=False, compile_model=False,
        )
        for i, row in enumerate(history, start=1):
            assert row["epoch"] == i, f"Expected epoch {i}, got {row['epoch']}"

    def test_elapsed_sec_present_every_epoch(self, tiny_model, tiny_loader, device):
        optimizer = torch.optim.SGD(tiny_model.parameters(), lr=0.01)
        history = train(
            tiny_model, optimizer, tiny_loader, tiny_loader,
            nn.CrossEntropyLoss(), device, epochs=2, verbose=False, compile_model=False,
        )
        for i, row in enumerate(history):
            assert "elapsed_sec" in row, f"elapsed_sec missing from epoch {i+1}"
            assert row["elapsed_sec"] > 0.0, f"elapsed_sec <= 0 at epoch {i+1}"


# ---------------------------------------------------------------------------
# divergence_rate
# ---------------------------------------------------------------------------

class TestDivergenceRate:
    def test_positive_when_test_greater_than_train(self):
        assert divergence_rate(0.3, 0.5) == pytest.approx(0.2)

    def test_zero_when_equal(self):
        assert divergence_rate(0.4, 0.4) == pytest.approx(0.0)

    def test_negative_when_test_less_than_train(self):
        assert divergence_rate(0.5, 0.3) == pytest.approx(-0.2)


# ---------------------------------------------------------------------------
# aggregate_seeds
# ---------------------------------------------------------------------------

class TestAggregateSeeds:
    SEED_METRICS = [
        {"test_acc": 0.90, "test_loss": 0.30, "train_acc": 0.99, "train_loss": 0.05,
         "divergence_rate": 0.25, "elapsed_sec": 10.0},
        {"test_acc": 0.92, "test_loss": 0.28, "train_acc": 0.99, "train_loss": 0.04,
         "divergence_rate": 0.24, "elapsed_sec": 11.0},
        {"test_acc": 0.91, "test_loss": 0.29, "train_acc": 1.00, "train_loss": 0.03,
         "divergence_rate": 0.26, "elapsed_sec": 10.5},
    ]

    EXPECTED_AGGREGATE_KEYS = {
        "test_acc_mean", "test_acc_sem",
        "test_loss_mean", "test_loss_sem",
        "train_acc_mean", "train_acc_sem",
        "train_loss_mean", "train_loss_sem",
        "divergence_rate_mean", "divergence_rate_sem",
        "elapsed_sec_mean", "elapsed_sec_sem",
    }

    def test_all_aggregate_keys_present(self):
        agg = aggregate_seeds(self.SEED_METRICS)
        missing = self.EXPECTED_AGGREGATE_KEYS - set(agg.keys())
        assert not missing, f"aggregate_seeds missing keys: {missing}"

    def test_mean_values_correct(self):
        agg = aggregate_seeds(self.SEED_METRICS)
        assert agg["test_acc_mean"] == pytest.approx((0.90 + 0.92 + 0.91) / 3, rel=1e-5)

    def test_sem_is_non_negative(self):
        agg = aggregate_seeds(self.SEED_METRICS)
        for key in agg:
            if key.endswith("_sem"):
                assert agg[key] >= 0.0, f"{key} is negative"

    def test_single_seed_sem_is_zero(self):
        agg = aggregate_seeds([self.SEED_METRICS[0]])
        for key in agg:
            if key.endswith("_sem"):
                assert agg[key] == 0.0, f"Single-seed {key} should be 0"
