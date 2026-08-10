"""Write the authoritative train/test split into every body's parameter.json.

Fixes a stale marker. Commit 9495329 put `"split": "val"` on `pear_shaped` and
`teen` when the 11 new bodies were generated, but NOTHING has ever read that
field -- and the split chosen from the Stage 0 actuator audit
(scripts/audit_bodies.py) puts both of those bodies in TRAIN. A dead field that
contradicts the live configuration is exactly the kind of thing that gets
believed years later, so it is made live here rather than left to rot.

After this runs, `parameter.json["split"]` is one of:
    "train"   in BilevelConfig.train_bodies
    "test"    in BilevelConfig.heldout_bodies
    "unused"  neither (currently only `heavy`, see config.py for why)

model/bilevel/data.py:load_body asserts the file agrees with the config, so the
two cannot drift apart again silently.

Run (after changing the split in model/bilevel/config.py):
    uv run scripts/write_body_splits.py
    uv run scripts/write_body_splits.py --robots assets/robots      # originals too
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.bilevel.config import BilevelConfig  # noqa: E402
from model.bilevel.data import split_of  # noqa: E402


def main():
    cfg = BilevelConfig()
    ap = argparse.ArgumentParser()
    ap.add_argument("--robots", type=Path, default=REPO_ROOT / cfg.robots_dir)
    args = ap.parse_args()

    train, test = set(cfg.train_bodies), set(cfg.heldout_bodies)
    overlap = train & test
    if overlap:
        raise SystemExit(f"a body cannot be in both train and test: {sorted(overlap)}")

    bodies = sorted(p for p in args.robots.iterdir() if (p / "parameter.json").exists())
    known = {p.name for p in bodies}
    missing = (train | test) - known
    if missing:
        raise SystemExit(f"config names bodies that do not exist in {args.robots}: {sorted(missing)}")

    print(f"{'body':<14}{'was':>10}{'now':>10}")
    print("-" * 34)
    for body in bodies:
        path = body / "parameter.json"
        par = json.loads(path.read_text())
        was = par.get("split", "-")
        now = split_of(cfg, body.name)   # one definition, shared with data.load_body
        par["split"] = now
        path.write_text(json.dumps(par, indent=2) + "\n")
        flag = "  <- changed" if was != now else ""
        print(f"{body.name:<14}{was:>10}{now:>10}{flag}")

    n_unused = len(known - train - test - {cfg.source_body})
    print(f"\n{len(train)} train / {len(test)} test / 1 source ({cfg.source_body}) / "
          f"{n_unused} unused  ->  {args.robots}")


if __name__ == "__main__":
    main()
