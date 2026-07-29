"""Entry point for launching training with the default config on GPU.

Usage (from project root):
    uv run model/run_train.py
    uv run model/run_train.py --gpu 1      # run on cuda:1
    uv run model/run_train.py --device cpu # or any explicit device string
"""
import argparse
import sys
from pathlib import Path

# Running this file directly (not `-m model.run_train`) puts model/ itself on
# sys.path, not its parent -- so `model.config`/`model.train`'s absolute
# imports can't find the `model` package unless the repo root is added too.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.config import TrainConfig
from model.train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index, e.g. 1 for cuda:1")
    parser.add_argument("--device", default=None,
                         help="explicit device string (e.g. 'cpu', 'cuda:2'); overrides --gpu")
    args = parser.parse_args()

    device = args.device or f"cuda:{args.gpu}"
    train(TrainConfig(device=device))


if __name__ == "__main__":
    main()
