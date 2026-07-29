"""Entry point for launching training with the default config on GPU.

Usage (from project root):
    uv run model/run_train.py
"""
import sys
from pathlib import Path

# Running this file directly (not `-m model.run_train`) puts model/ itself on
# sys.path, not its parent -- so `model.config`/`model.train`'s absolute
# imports can't find the `model` package unless the repo root is added too.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.config import TrainConfig
from model.train import train

if __name__ == "__main__":
    train(TrainConfig(device="cuda:0"))
