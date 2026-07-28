"""Dump the rest-pose (qpos=0) world-space transform of every body in a MJCF
humanoid, plus its geoms, to a plain JSON file.

Building the corresponding Blender armature from raw MJCF <body pos>/<quat>
chains would mean re-implementing MuJoCo's quaternion composition/normalization
rules by hand. Instead we let MuJoCo itself compute the authoritative rest-pose
world transform for every body (mj_forward at qpos=0), and only the resulting
flat world positions/orientations get handed to Blender. This keeps the MJCF
parsing decoupled from any assumptions about how Blender will build the rig.
"""

import argparse
import json

import mujoco
import numpy as np


def body_parent_name(model, body_id):
    parent_id = model.body_parentid[body_id]
    if parent_id == body_id:  # worldbody is its own parent
        return None
    return model.body(parent_id).name


def dump_skeleton(input_path, output_path):
    model = mujoco.MjModel.from_xml_path(input_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)  # qpos defaults to 0 -> rest pose

    bodies = []
    for body_id in range(model.nbody):
        name = model.body(body_id).name
        if name == "world":
            continue
        parent = body_parent_name(model, body_id)
        bodies.append({
            "name": name,
            "parent": parent,
            "world_pos": data.xpos[body_id].tolist(),
            "world_quat_wxyz": data.xquat[body_id].tolist(),
            "mass": float(model.body_mass[body_id]),
        })

    geoms = []
    for geom_id in range(model.ngeom):
        body_id = model.geom_bodyid[geom_id]
        body_name = model.body(body_id).name
        if body_name == "world":
            continue
        geom_type = mujoco.mjtGeom(model.geom_type[geom_id]).name
        geoms.append({
            "name": model.geom(geom_id).name,
            "body": body_name,
            "type": geom_type,
            "size": model.geom_size[geom_id].tolist(),
            "local_pos": model.geom_pos[geom_id].tolist(),
            "local_quat_wxyz": model.geom_quat[geom_id].tolist(),
            "world_pos": data.geom_xpos[geom_id].tolist(),
            "world_quat_wxyz": _mat_to_quat(data.geom_xmat[geom_id]),
        })

    with open(output_path, "w") as f:
        json.dump({"bodies": bodies, "geoms": geoms}, f, indent=2)
    print(f"Wrote {len(bodies)} bodies, {len(geoms)} geoms -> {output_path}")


def _mat_to_quat(xmat_flat):
    mat = np.array(xmat_flat).reshape(3, 3)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat.flatten())
    return quat.tolist()


def main():
    parser = argparse.ArgumentParser(description="Dump MJCF rest-pose world transforms to JSON.")
    parser.add_argument("--input", default="assets/robots/robot.xml")
    parser.add_argument("--output", default="assets/robots/robot.json")
    args = parser.parse_args()
    dump_skeleton(args.input, args.output)


if __name__ == "__main__":
    main()
