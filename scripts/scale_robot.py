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

Writes assets/robots/robot_<label>.xml and assets/robots/robot_<label>_parameter.json
(or assets/robots/robot_parameter.json for the baseline, since robot.xml already exists).
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


def _local_scale(params, group):
    """Scale factor for a joint's own local rotational inertia after
    scaling: mass of a limb segment scales ~ length * girth^2 (volume,
    density held constant -- see the per-geom `density=` attrs in robot.xml,
    no explicit <inertial> override, so MuJoCo derives mass/inertia straight
    from geom size), and inertia = mass * radius^2 with radius scaling ~
    length, so local inertia scales ~ length^3 * girth^2. Used for
    `armature`/joint `damping`/`stiffness` -- these represent the joint's
    OWN passive/rotor dynamics, not the load it's supporting (see
    compute_joint_loads for that), so they should track the joint's own
    segment shrinking, not what it's holding up."""
    length_scale = params[f"{group}_scale"]
    girth_scale = params[f"{group}_girth"]
    return (length_scale ** 3) * (girth_scale ** 2)


def compute_joint_loads(xml_string, leg_joint_names):
    """Per-joint 'load' = mass supported * lever arm from that mass's center
    of mass to the joint's anchor point -- an approximation of the static
    gravity torque this joint must supply. Uses MuJoCo's own
    body_subtreemass/subtree_com (auto-derived from geom size + density, so
    it reflects the ACTUAL scaled geometry, not a hand-derived volume
    formula). Leg joints are treated specially: during stance phase a single
    leg supports close to the WHOLE body's weight, not just its own subtree,
    so their load uses max(own_subtree_mass, total_mass - own_subtree_mass)
    at the whole-body center of mass -- everything else (arms, torso, neck,
    head) just uses its own subtree's mass/COM, i.e. what's actually distal
    to that joint.

    This replaces a per-body-GROUP torque scale (length_scale^2 *
    girth_scale^2) that assumed a joint only ever supports its own limb
    segment's weight -- wrong for any joint whose real load comes from
    elsewhere in the chain (legs supporting the whole body; the neck, grouped
    under "torso", actually supporting the head, which can be scaled
    differently/oppositely from the torso -- e.g. the child preset shrinks
    torso but grows the head)."""
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    root_id = model.body("Pelvis").id
    total_mass = model.body_subtreemass[root_id]
    root_com = data.subtree_com[root_id]

    loads = {}
    for jid in range(model.njnt):
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        body_id = model.jnt_bodyid[jid]
        anchor = data.xanchor[jid]
        if name in leg_joint_names:
            own = model.body_subtreemass[body_id]
            mass = max(own, total_mass - own)
            com = root_com
        else:
            mass = model.body_subtreemass[body_id]
            com = data.subtree_com[body_id]
        lever = np.linalg.norm(com - anchor)
        loads[name] = mass * max(lever, 1e-3)
    return loads


_DEFAULT_JOINT_STIFFNESS = {None: 2.0, "stiff_medium": 10.0, "stiff_medium_higher": 50.0, "stiff_high": 100.0}
_DEFAULT_JOINT_ARMATURE = 0.01


def _resolve_stiffness(joint):
    """Effective stiffness for a <joint>: explicit attr > class lookup >
    top-level <default><joint> value. robot.xml sets stiffness either
    directly on the joint, via class="stiff_medium/_higher/_high", or leaves
    it to the top-level default (2.0) -- resolve whichever applies so it can
    be scaled and written back explicitly."""
    if joint.get("stiffness") is not None:
        return float(joint.get("stiffness"))
    return _DEFAULT_JOINT_STIFFNESS[joint.get("class")]


def apply_scale(tree, params, source_xml=SOURCE_XML):
    root = tree.getroot()

    n_geoms_touched = 0
    for body in root.iter("body"):
        group = GROUP_OF_BODY.get(body.get("name"))
        if group is None:
            continue  # Pelvis (root) and anything ungrouped stay untouched
        length_scale = params[f"{group}_scale"]
        girth_scale = params[f"{group}_girth"]

        body.set("pos", _scale_floats(body.get("pos"), length_scale))

        for joint in body.findall("joint"):
            # Every joint in robot.xml is at pos="0 0 0" (relative to its
            # body's own origin) -- load-bearing for the per-joint load/
            # inertia scaling below, which only rescales body/geom pos, not
            # joint pos. Fail loudly if a future skeleton edit breaks this.
            jpos = joint.get("pos")
            assert jpos is None or tuple(float(v) for v in jpos.split()) == (0.0, 0.0, 0.0), (
                f"joint {joint.get('name')!r} has non-zero pos={jpos!r} -- "
                "apply_scale doesn't rescale joint pos, only body/geom pos"
            )

        for geom in body.findall("geom"):
            if geom.get("fromto") is not None:
                geom.set("fromto", _scale_floats(geom.get("fromto"), length_scale))
            elif geom.get("pos") is not None:
                geom.set("pos", _scale_floats(geom.get("pos"), length_scale))
            if geom.get("size") is not None:
                geom.set("size", _scale_floats(geom.get("size"), girth_scale))
                n_geoms_touched += 1

    # Pelvis's own contact box has no group -- follow torso girth.
    for body in root.iter("body"):
        if body.get("name") == "Pelvis":
            for geom in body.findall("geom"):
                if geom.get("size") is not None:
                    geom.set("size", _scale_floats(geom.get("size"), params["torso_girth"]))
                    n_geoms_touched += 1

    # root.iter("geom") also walks <default><geom .../> template elements
    # (not real instantiated geoms) -- only count geoms that are actual
    # <body> descendants. "floor" (in worldbody) is deliberately excluded:
    # it should never be scaled.
    body_geom_ids = {id(g) for b in root.iter("body") for g in b.findall("geom")}
    real_geoms = [g for g in root.iter("geom") if id(g) in body_geom_ids]
    n_untouched = len(real_geoms) - n_geoms_touched
    if n_untouched:
        print(f"NOTE: {n_untouched}/{len(real_geoms)} body geoms have no `size` attr "
              f"(e.g. inherit from a <default> class) and were not scaled -- "
              f"verify this is expected.")

    joint_to_body_name = {}
    for body in root.iter("body"):
        body_name = body.get("name")
        for joint in body.findall("joint"):
            joint_to_body_name[joint.get("name")] = body_name
    leg_joint_names = {name for name, body_name in joint_to_body_name.items()
                        if GROUP_OF_BODY.get(body_name) == "leg"}

    # Joint dynamics (armature/damping/stiffness): each joint's OWN local
    # inertia scale (length^3 * girth^2 of its own body group) -- these
    # represent the joint's passive/rotor response, not the load it
    # supports, so they track the joint's own segment shrinking. Preserves
    # natural frequency sqrt(stiffness/inertia) and damping ratio
    # damping/(2*sqrt(stiffness*inertia)) as inertia scales, instead of the
    # joint becoming disproportionately stiff/underdamped as it shrinks.
    # armature has no per-joint override in robot.xml (only the top-level
    # <default><joint armature="0.01">), so this adds an explicit override.
    for body in root.iter("body"):
        group = GROUP_OF_BODY.get(body.get("name"))
        if group is None:
            continue
        local_scale = _local_scale(params, group)
        for joint in body.findall("joint"):
            stiffness = _resolve_stiffness(joint) * local_scale
            joint.set("stiffness", f"{stiffness:.6g}")
            if joint.get("damping") is not None:
                joint.set("damping", f"{float(joint.get('damping')) * local_scale:.6g}")
            armature = float(joint.get("armature", _DEFAULT_JOINT_ARMATURE)) * local_scale
            joint.set("armature", f"{armature:.6g}")

    # Actuators: find each <general> actuator's owning joint, then scale
    # gain/bias/force-limit by that joint's OWN load ratio (see
    # compute_joint_loads) -- the mass+lever-arm it actually has to support,
    # not just its own body group's local inertia. Previously left untouched
    # entirely (an exact copy of robot.xml's actuators regardless of body
    # size), then briefly scaled by a per-body-GROUP heuristic that ignored
    # cross-body load transfer (legs carrying the whole body during stance,
    # the neck carrying the head) -- both empirically made the child body
    # fall far more than the original under the same frozen policy + z.
    original_xml_string = Path(source_xml).read_text()
    loads_orig = compute_joint_loads(original_xml_string, leg_joint_names)
    scaled_geometry_xml_string = ET.tostring(root, encoding="unicode")
    loads_new = compute_joint_loads(scaled_geometry_xml_string, leg_joint_names)

    actuator_root = root.find("actuator")
    if actuator_root is not None:
        clamp_lo, clamp_hi = 0.02, 5.0
        scale_report = []
        for act in actuator_root.findall("general"):
            joint_name = act.get("joint")
            if joint_name not in loads_orig:
                continue  # no group (shouldn't happen for the 69 hinge actuators)
            raw_scale = loads_new[joint_name] / loads_orig[joint_name]
            scale = min(max(raw_scale, clamp_lo), clamp_hi)
            scale_report.append((joint_name, raw_scale, scale))
            if raw_scale != scale:
                print(f"WARNING: {joint_name} torque scale {raw_scale:.4f} clamped to {scale:.4f}")

            if act.get("gainprm") is not None:
                act.set("gainprm", _scale_floats(act.get("gainprm"), scale))
            if act.get("biasprm") is not None:
                act.set("biasprm", _scale_floats(act.get("biasprm"), scale))
            if act.get("forcerange") is not None:
                act.set("forcerange", _scale_floats(act.get("forcerange"), scale))

        scale_report.sort(key=lambda r: r[2])
        print("per-joint actuator torque scale (lowest/highest 5):")
        for name, raw, clamped in scale_report[:5]:
            print(f"  {name:16s} {clamped:.4f}" + (f"  (raw {raw:.4f})" if raw != clamped else ""))
        print("  ...")
        for name, raw, clamped in scale_report[-5:]:
            print(f"  {name:16s} {clamped:.4f}" + (f"  (raw {raw:.4f})" if raw != clamped else ""))


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
    apply_scale(tree, params, source_xml=source_xml)

    # Feet exactly flush with the ground (gap = 0) in the default pose.
    xml_string = ET.tostring(tree.getroot(), encoding="unicode")
    dz = -measure_min_z(xml_string)
    for body in tree.getroot().iter("body"):
        if body.get("name") == "Pelvis":
            x, y, z = (float(v) for v in body.get("pos").split())
            body.set("pos", f"{x:.6g} {y:.6g} {z + dz:.6g}")

    out_xml = out_dir / f"robot_{label}.xml"
    tree.write(out_xml)

    # NOT robot_{label}.json -- that name is owned by the skeleton-export
    # schema (bodies/world_pos/mass list, from
    # mujoco_Humanoid_motionretargeting/scripts/export_skeleton_json.py),
    # a completely different file consumed by qpos_retarget.py. This is the
    # flat scale-factor schema instead (see model/dataset.py's BETA_AXES).
    out_json = out_dir / f"robot_{label}_parameter.json"
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
        # (identity) morphology params next to it. NOT robot.json -- see the
        # comment above generate()'s out_json, same schema collision.
        out_json = ROBOTS_DIR / "robot_parameter.json"
        out_json.write_text(json.dumps({"label": "baseline", **params}, indent=2) + "\n")
        print(f"wrote {out_json} (robot.xml left untouched)")
        return

    generate(label, params)


if __name__ == "__main__":
    main()
