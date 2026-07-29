"""
Enumerate reward-name combinations for humenv, given a chosen discretization
grid for each continuous parameter. See docs/humenv_reward_functions.md for
the full pattern reference and the official ALL_TASKS grid.

Edit the GRIDS dict below to widen/narrow the search space, then run
(from the project root):
    uv run scripts/humenv_reward_combinations.py

Writes docs/humenv_reward_combinations.txt.
"""

import itertools
from pathlib import Path

OUT_PATH = Path(__file__).resolve().parent.parent / "docs" / "humenv_reward_combinations.txt"

ARM_POSES = ["l", "m", "h", "x"]

# Adjust these grids to control how many combinations are generated.
GRIDS = {
    "move-ego": {
        "angle": [0, -90, 90, 180],
        "speed": [2, 4],
    },
    "move-ego-low": {
        "angle": [0, -90, 90, 180],
        "speed": [2],
    },
    "jump": {
        "height": [1.0, 1.6, 2.0],
    },
    "rotate": {
        "axis": ["x", "y", "z"],
        "ang_vel": [-5, 5],
        "pelvis_height": [0.8],
    },
    "raisearms": {
        "left": ARM_POSES,
        "right": ARM_POSES,
    },
    "split": {
        "distance": [0.5, 1.0, 1.5],
    },
    "crouch": {
        "height_th": [0, 0.3, 0.6],
    },
    "crawl": {
        "speed": [0, 2],
        "angle": [0, 180],
        "direction": ["u", "d"],
    },
    "move-ego-raisearms": {
        "angle": [0],
        "speed": [2],
        "left": ARM_POSES,
        "right": ARM_POSES,
    },
}

FIXED_NAMES = ["headstand", "sitonground", "lieonground-up", "lieonground-down"]


def build_names():
    names = {}

    g = GRIDS["move-ego"]
    names["move-ego"] = [f"move-ego-{a}-{s}" for a, s in itertools.product(g["angle"], g["speed"])]

    g = GRIDS["move-ego-low"]
    names["move-ego-low"] = [f"move-ego-low-{a}-{s}" for a, s in itertools.product(g["angle"], g["speed"])]

    g = GRIDS["jump"]
    names["jump"] = [f"jump-{h}" for h in g["height"]]

    g = GRIDS["rotate"]
    names["rotate"] = [
        f"rotate-{ax}-{av}-{ph}"
        for ax, av, ph in itertools.product(g["axis"], g["ang_vel"], g["pelvis_height"])
    ]

    g = GRIDS["raisearms"]
    names["raisearms"] = [f"raisearms-{l}-{r}" for l, r in itertools.product(g["left"], g["right"])]

    g = GRIDS["split"]
    names["split"] = [f"split-{d}" for d in g["distance"]]

    g = GRIDS["crouch"]
    names["crouch"] = [f"crouch-{h}" for h in g["height_th"]]

    g = GRIDS["crawl"]
    names["crawl"] = [
        f"crawl-{sp}-{a}-{d}" for sp, a, d in itertools.product(g["speed"], g["angle"], g["direction"])
    ]

    g = GRIDS["move-ego-raisearms"]
    names["move-ego-raisearms"] = [
        f"move-ego-{a}-{s}-raisearms-{l}-{r}"
        for a, s, l, r in itertools.product(g["angle"], g["speed"], g["left"], g["right"])
    ]

    names["fixed"] = FIXED_NAMES
    return names


def main():
    names = build_names()
    total = 0
    for group, group_names in names.items():
        print(f"{group:22s}: {len(group_names)}")
        total += len(group_names)
    print("-" * 34)
    print(f"{'TOTAL':22s}: {total}")

    all_names = [n for group_names in names.values() for n in group_names]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write("\n".join(all_names) + "\n")
    print(f"\nWrote {len(all_names)} reward names to {OUT_PATH}")


if __name__ == "__main__":
    main()
