"""Behavior checks for reproducibility, logging, checkpoints and factory contracts."""

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from od_scratch.infra.checkpoint import load_checkpoint, save_checkpoint
from od_scratch.infra.config import RunConfig, load_config
from od_scratch.infra.device import memory_stats, select_device
from od_scratch.infra.experiment import Experiment
from od_scratch.infra.optim import make_optimizer, make_scheduler
from od_scratch.infra.profiling import Timer, WeightedMean, gradient_norms, parameter_counts
from od_scratch.infra.reproducibility import seed_everything


def test_seed_repeats_all_rngs() -> None:
    seed_everything(17)
    first = (random.random(), np.random.rand(), torch.rand(4))
    seed_everything(17)
    second = (random.random(), np.random.rand(), torch.rand(4))
    assert first[:2] == second[:2]
    assert torch.equal(first[2], second[2])


def test_config_rejects_typo_and_invalid_batch(tmp_path: Path) -> None:
    path = tmp_path / "run.toml"
    path.write_text("batch_size = 3", encoding="utf-8")
    assert load_config(path).batch_size == 3
    with pytest.raises(ValidationError):
        RunConfig(batch_size=0)
    with pytest.raises(ValidationError):
        RunConfig.model_validate({"batch_szie": 3})


def test_device_contract() -> None:
    assert select_device("cpu").type == "cpu"
    assert memory_stats(torch.device("cpu")) == {}
    with pytest.raises(ValueError):
        select_device("invalid")


def test_checkpoint_roundtrip_optimizer(tmp_path: Path) -> None:
    model = torch.nn.Linear(3, 2)
    optimizer = make_optimizer(model.parameters())
    model(torch.ones(1, 3)).sum().backward()
    optimizer.step()
    expected = {key: value.clone() for key, value in model.state_dict().items()}
    path = tmp_path / "nested" / "state.pt"
    save_checkpoint(path, model, training_state={"epoch": 1, "optimizer": optimizer.state_dict()})
    with torch.no_grad():
        model.weight.zero_()
    state = load_checkpoint(path, model)
    optimizer.load_state_dict(state["optimizer"])
    assert state["epoch"] == 1
    assert all(torch.equal(model.state_dict()[key], value) for key, value in expected.items())


def test_checkpoint_rejects_head_shape(tmp_path: Path) -> None:
    path = tmp_path / "state.pt"
    save_checkpoint(path, torch.nn.Linear(3, 2))
    with pytest.raises(RuntimeError):
        load_checkpoint(path, torch.nn.Linear(3, 4), strict=False)


def test_experiment_json_and_tensorboard(tmp_path: Path) -> None:
    config = RunConfig(artifact_root=str(tmp_path))
    with Experiment(config) as run:
        run.log(2, {"loss/localization": 1.25})
        with pytest.raises(ValueError):
            run.log(3, {"loss": float("nan")})
    event = json.loads((run.path / "metrics.jsonl").read_text())
    assert event["metrics"]["loss/localization"] == 1.25
    accumulator = EventAccumulator(str(run.path / "tensorboard"))
    accumulator.Reload()
    assert accumulator.Scalars("loss/localization")[0].step == 2
    with Experiment(config, tensorboard=False) as other:
        assert other.path != run.path


def test_gradients_and_frozen_parameters() -> None:
    model = torch.nn.Linear(3, 2)
    model.bias.requires_grad_(False)
    model(torch.ones(1, 3)).sum().backward()
    assert parameter_counts(model) == {"total": 8, "trainable": 6}
    norms = gradient_norms(model)
    assert "grad_norm/bias" not in norms
    assert norms["grad_norm/total"] == pytest.approx(6**0.5)


def test_scheduler_respects_budget_and_parameter_groups() -> None:
    backbone = torch.nn.Parameter(torch.ones(1))
    head = torch.nn.Parameter(torch.ones(1))
    optimizer = make_optimizer([{"params": [backbone], "lr": 1e-4}, {"params": [head], "lr": 1e-3}])
    scheduler = make_scheduler(optimizer, "cosine", epochs=4)
    assert scheduler is not None
    for _ in range(4):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0)
    with pytest.raises(ValueError):
        make_scheduler(optimizer, "typo", epochs=4)
    with pytest.raises(ValueError):
        make_optimizer([backbone], "typo")


def test_weighted_statistics_and_timer() -> None:
    mean = WeightedMean()
    with pytest.raises(ValueError):
        _ = mean.mean
    mean.update(2, 3)
    mean.update(8, 1)
    assert mean.mean == 3.5
    with Timer(torch.device("cpu")) as timer:
        torch.ones(2).sum()
    assert timer.elapsed >= 0
