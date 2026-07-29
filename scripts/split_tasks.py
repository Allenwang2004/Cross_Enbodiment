"""Split the official HumEnv task list (docs/humenv_all_tasks_official.txt)
into train/test task sets.

Split at the *task* (reward_name) level, not the row level -- all 10 trials
of a given reward_name carry highly correlated motion (same z sampling
distribution), so splitting individual rows would leak near-duplicate
motion across train/test and inflate eval scores.

Usage (from project root):
    uv run scripts/split_tasks.py
    uv run scripts/split_tasks.py --test-frac 0.2 --seed 0
"""

import argparse
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TASKS_FILE = REPO_ROOT / "docs" / "humenv_all_tasks_official.txt"
SPLIT_DIR = REPO_ROOT / "datasets" / "crossenbodiment-1-datasets" / "splits"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tasks = [line.strip() for line in TASKS_FILE.read_text().splitlines() if line.strip()]
    rng = random.Random(args.seed)
    shuffled = tasks[:]
    rng.shuffle(shuffled)

    n_test = max(1, round(len(shuffled) * args.test_frac))
    test_tasks = sorted(shuffled[:n_test])
    train_tasks = sorted(shuffled[n_test:])

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    (SPLIT_DIR / "train_tasks.txt").write_text("\n".join(train_tasks) + "\n")
    (SPLIT_DIR / "test_tasks.txt").write_text("\n".join(test_tasks) + "\n")

    print(f"{len(train_tasks)} train tasks, {len(test_tasks)} test tasks -> {SPLIT_DIR}")
    print("test tasks:", test_tasks)


if __name__ == "__main__":
    main()
