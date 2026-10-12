"""Infrastructure smoke check using tensor arithmetic, not a detector."""

import argparse
from pathlib import Path

import torch

from od_scratch.infra.config import load_config
from od_scratch.infra.device import memory_stats, select_device
from od_scratch.infra.experiment import Experiment, console_logger
from od_scratch.infra.profiling import Timer
from od_scratch.infra.reproducibility import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.toml"))
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(config.seed, deterministic=config.deterministic)
    device = select_device(config.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with Experiment(config) as experiment:
        with Timer(device) as timer:
            x = torch.randn(32, 32, device=device, requires_grad=True)
            loss = x.square().mean()
            torch.autograd.backward(loss)
        experiment.log(
            0,
            {
                "smoke/loss": float(loss.detach().item()),
                "smoke/elapsed_seconds": timer.elapsed,
                **memory_stats(device),
            },
        )
        console_logger().info("Smoke check passed on %s; run: %s", device, experiment.path)
