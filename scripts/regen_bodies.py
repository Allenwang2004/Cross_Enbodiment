"""Regenerate the body variants WITH load-based actuator scaling (Stage 0 / R1).

The 11 bodies added in commit 9495329 were generated with --no-actuator-scale,
so they carry the adult's gainprm/biasprm/forcerange despite masses from 38 kg
to 210 kg. scripts/audit_bodies.py measures the consequence: `heavy`,
`short_stocky` and `short_limbed` cannot produce enough torque to hold their own
REST pose (headroom 0.026 / 0.035 / 0.092 -- they need 10-40x what they have),
and `giant` and `pear_shaped` are also under 1.0. Bodies that cannot stand
still contribute nothing but gradient noise.

scripts/scale_robot.py already contains the fix: `compute_joint_loads` (:103)
scales each actuator by the ratio of the per-joint load (mass x lever arm, and
for leg joints max(own_subtree, total - own_subtree) about the whole-body CoM)
between the new body and the adult. It just was not applied.

This writes to a SEPARATE output directory by default rather than overwriting
assets/robots/, because regenerating invalidates comparisons against the
existing model/checkpoints/ and outputs/{baseline,eval}/report.json numbers.
Point BilevelConfig.robots_dir at the new directory once the audit passes.

Run:
    uv run scripts/regen_bodies.py                       # -> assets/robots_scaled/
    uv run scripts/regen_bodies.py --out assets/robots   # overwrite, deliberate
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MUJOCO_GL", "egl")

from scale_robot import AXES, generate  # noqa: E402  (scripts/ is not a package)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "assets" / "robots_scaled")
    ap.add_argument("--src", type=Path, default=REPO_ROOT / "assets" / "robots")
    ap.add_argument("--source-body", default="adult")
    args = ap.parse_args()

    src_xml = args.src / args.source_body / "robot.xml"
    args.out.mkdir(parents=True, exist_ok=True)

    bodies = sorted(p for p in args.src.iterdir() if (p / "parameter.json").exists())
    print(f"regenerating {len(bodies)} bodies from {src_xml} with load-based actuator scaling\n")

    for body in bodies:
        params = json.loads((body / "parameter.json").read_text())
        axes = {a: params[a] for a in AXES}
        if body.name == args.source_body:
            # The source body IS the reference the load ratios are taken against,
            # so rescaling it is a no-op by construction -- copy it verbatim.
            dst = args.out / body.name
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(body / "robot.xml", dst / "robot.xml")
            (dst / "parameter.json").write_text(
                json.dumps({"label": body.name, "scale_actuators": True, **axes}, indent=2) + "\n"
            )
            print(f"copied {body.name} (source body, actuators are the reference)")
            continue
        generate(body.name, axes, source_xml=src_xml, out_dir=args.out, scale_actuators=True)

    # skeleton.json is only consumed by the legacy scripts/qpos_retarget.py path
    # (model/bilevel reads rest height from MjModel.qpos0 instead), but copy any
    # that exist so the old scripts keep working against the new assets.
    for body in bodies:
        s = body / "skeleton.json"
        if s.exists():
            shutil.copy2(s, args.out / body.name / "skeleton.json")

    print(f"\nwrote {args.out}")
    print(f"re-audit with:  uv run scripts/audit_bodies.py --robots {args.out}")


if __name__ == "__main__":
    main()
