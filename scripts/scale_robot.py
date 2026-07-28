"""
Generate a scaled variant of assets/robots/robot.xml with independent
per-body-group scale factors: legs / arms / torso / head, each with its own
length scale and girth (thickness) scale. Body/joint/actuator names and gear
ratios are untouched, so the pretrained Meta Motivo model still works on the
result (same obs/action size).

After scaling, the Pelvis root height is auto-corrected via one MuJoCo
forward-kinematics pass so the feet still rest at ground level (z=0) in the
default pose -- otherwise a taller/shorter skeleton would float or clip
through the floor.

Presets:
    baseline  -- identity (scale=1.0 everywhere). Doesn't touch robot.xml
                 itself, just writes robot.json with the identity params so
                 every body variant has a matching metadata file.
    child     -- proportionally shorter limbs/torso, slightly bigger head,
                 thinner limbs.
    elderly   -- slightly shorter stature, thinner limbs, thicker torso.
These are rough illustrative defaults -- override any axis from the CLI.

Usage (from project root):
    uv run scripts/scale_robot.py --preset baseline
    uv run scripts/scale_robot.py --preset child
    uv run scripts/scale_robot.py --preset elderly
    uv run scripts/scale_robot.py --label custom --leg-scale 1.1 --leg-girth 0.9

Writes assets/robots/robot_<label>.xml and assets/robots/robot_<label>.json
(or assets/robots/robot.json for the baseline, since robot.xml already exists).
"""

import argparse
import itertools
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
ROBOTS_DIR = REPO_ROOT / "assets" / "robots"
SOURCE_XML = ROBOTS_DIR / "robot.xml"

BODY_GROUPS = {
    "leg": ["L_Hip", "L_Knee", "L_Ankle", "L_Toe", "R_Hip", "R_Knee", "R_Ankle", "R_Toe"],
    "arm": ["L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand",
            "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand"],
    "torso": ["Torso", "Spine", "Chest", "Neck"],
    "head": ["Head"],
}
GROUP_OF_BODY = {name: group for group, names in BODY_GROUPS.items() for name in names}

AXES = ["leg_scale", "arm_scale", "torso_scale", "head_scale",
        "leg_girth", "arm_girth", "torso_girth", "head_girth"]

PRESETS = {
    "baseline": dict(leg_scale=1.00, arm_scale=1.00, torso_scale=1.00, head_scale=1.00,
                      leg_girth=1.00, arm_girth=1.00, torso_girth=1.00, head_girth=1.00),
    # Children: proportionally shorter legs/arms/torso, slightly bigger &
    # rounder head, thinner limbs.
    "child": dict(leg_scale=0.62, arm_scale=0.65, torso_scale=0.75, head_scale=1.05,
                  leg_girth=0.70, arm_girth=0.70, torso_girth=0.85, head_girth=1.15),
    # Elderly: mild height loss, thinner limbs (sarcopenia), thicker torso.
    "elderly": dict(leg_scale=0.92, arm_scale=0.97, torso_scale=0.95, head_scale=1.00,
                     leg_girth=0.80, arm_girth=0.78, torso_girth=1.08, head_girth=1.00),
}


def _scale_floats(text, factor):
    return " ".join(f"{float(x) * factor:.6g}" for x in text.split())


def apply_scale(tree, params):
    root = tree.getroot()
    for body in root.iter("body"):
        group = GROUP_OF_BODY.get(body.get("name"))
        if group is None:
            continue  # Pelvis (root) and anything ungrouped stay untouched
        length_scale = params[f"{group}_scale"]
        girth_scale = params[f"{group}_girth"]

        body.set("pos", _scale_floats(body.get("pos"), length_scale))

        for geom in body.findall("geom"):
            if geom.get("fromto") is not None:
                geom.set("fromto", _scale_floats(geom.get("fromto"), length_scale))
            elif geom.get("pos") is not None:
                geom.set("pos", _scale_floats(geom.get("pos"), length_scale))
            if geom.get("size") is not None:
                geom.set("size", _scale_floats(geom.get("size"), girth_scale))

    # Pelvis's own contact box has no group -- follow torso girth.
    for body in root.iter("body"):
        if body.get("name") == "Pelvis":
            for geom in body.findall("geom"):
                if geom.get("size") is not None:
                    geom.set("size", _scale_floats(geom.get("size"), params["torso_girth"]))


def measure_min_z(xml_string):
    """Exact lowest point (z) of any non-floor geom in the default pose:
    box geoms via their 8 corners, capsules via their two cap-sphere
    centers minus radius, spheres via center minus radius. (geom_rbound is
    a bounding-SPHERE radius and is not tight for box geoms -- feet here
    are boxes -- so it under/over-estimates clearance depending on the
    box's aspect ratio, which changes non-uniformly once girth and length
    are scaled independently. Exact per-shape geometry avoids that.)"""
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    min_z = np.inf
    for gid in range(model.ngeom):
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) == "floor":
            continue
        gtype = model.geom_type[gid]
        xpos = data.geom_xpos[gid]
        xmat = data.geom_xmat[gid].reshape(3, 3)
        size = model.geom_size[gid]
        if gtype == mujoco.mjtGeom.mjGEOM_BOX:
            corners = [
                xpos + xmat @ (np.array(signs) * size)
                for signs in itertools.product([1, -1], repeat=3)
            ]
            z = min(c[2] for c in corners)
        elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
            c1 = xpos + xmat @ np.array([0, 0, size[1]])
            c2 = xpos + xmat @ np.array([0, 0, -size[1]])
            z = min(c1[2], c2[2]) - size[0]
        elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            z = xpos[2] - size[0]
        else:
            z = xpos[2]
        min_z = min(min_z, z)
    return min_z


def generate(label, params, source_xml=SOURCE_XML, out_dir=ROBOTS_DIR):
    tree = ET.parse(source_xml)
    apply_scale(tree, params)

    # Feet exactly flush with the ground (gap = 0) in the default pose.
    xml_string = ET.tostring(tree.getroot(), encoding="unicode")
    dz = -measure_min_z(xml_string)
    for body in tree.getroot().iter("body"):
        if body.get("name") == "Pelvis":
            x, y, z = (float(v) for v in body.get("pos").split())
            body.set("pos", f"{x:.6g} {y:.6g} {z + dz:.6g}")

    out_xml = out_dir / f"robot_{label}.xml"
    tree.write(out_xml)

    out_json = out_dir / f"robot_{label}.json"
    out_json.write_text(json.dumps({"label": label, **params}, indent=2) + "\n")

    print(f"wrote {out_xml}")
    print(f"wrote {out_json}")
    return out_xml, out_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=list(PRESETS), default=None)
    parser.add_argument("--label", default=None,
                         help="output name -> robot_<label>.xml/.json "
                              "(defaults to --preset name)")
    for axis in AXES:
        parser.add_argument(f"--{axis.replace('_', '-')}", type=float, default=None,
                             help=f"override this preset's {axis}")
    args = parser.parse_args()

    if args.preset is None and args.label is None:
        parser.error("pass --preset or --label")

    params = dict(PRESETS[args.preset] if args.preset else PRESETS["baseline"])
    label = args.label or args.preset
    for axis in AXES:
        cli_val = getattr(args, axis.replace("-", "_"))
        if cli_val is not None:
            params[axis] = cli_val

    if label == "baseline" and all(params[a] == 1.0 for a in AXES):
        # robot.xml already exists and is untouched -- just record its
        # (identity) morphology params next to it.
        out_json = ROBOTS_DIR / "robot.json"
        out_json.write_text(json.dumps({"label": "baseline", **params}, indent=2) + "\n")
        print(f"wrote {out_json} (robot.xml left untouched)")
        return

    generate(label, params)


if __name__ == "__main__":
    main()
