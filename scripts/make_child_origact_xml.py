"""Build a hybrid body: robot_child.xml's scaled-down geometry/inertia with
robot.xml's original (unscaled) actuator gains/force-ranges swapped in
wholesale, for the kinematics-only exploration experiment in
model/train_explore.py (see model.md / the plan for why -- short version:
isolate "can the child body's own morphology, driven by full adult actuator
strength, learn to stay physically plausible" from scale_robot.py's own
load-ratio actuator scaling).

Mechanically trivial because robot.xml and robot_child.xml share identical
actuator names/joint refs/declaration order (scale_robot.py never touches
body/joint/actuator naming or ordering, only numeric attributes) -- this
just swaps the whole <actuator> element, asserting that identity first.

Usage (from project root):
    uv run scripts/make_child_origact_xml.py

Writes assets/robots/child/robot_origact.xml.
"""

from pathlib import Path
import xml.etree.ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parent.parent
ROBOTS_DIR = REPO_ROOT / "assets" / "robots"
CHILD_XML = ROBOTS_DIR / "child" / "robot.xml"
ADULT_XML = ROBOTS_DIR / "adult" / "robot.xml"
OUT_XML = ROBOTS_DIR / "child" / "robot_origact.xml"


def _actuator_signature(actuator_root):
    return [(act.get("name"), act.get("joint")) for act in actuator_root.findall("general")]


def main():
    child_tree = ET.parse(CHILD_XML)
    adult_tree = ET.parse(ADULT_XML)
    child_root = child_tree.getroot()
    adult_root = adult_tree.getroot()

    child_actuator = child_root.find("actuator")
    adult_actuator = adult_root.find("actuator")
    assert child_actuator is not None and adult_actuator is not None, \
        "expected an <actuator> element in both files"

    child_sig = _actuator_signature(child_actuator)
    adult_sig = _actuator_signature(adult_actuator)
    assert child_sig == adult_sig, (
        f"actuator name/joint signature mismatch between {CHILD_XML.name} and "
        f"{ADULT_XML.name} -- expected identical names/joints/order, got "
        f"{len(child_sig)} vs {len(adult_sig)} actuators, first diff at "
        f"{next((i for i, (a, b) in enumerate(zip(child_sig, adult_sig)) if a != b), None)}"
    )

    child_root.remove(child_actuator)
    child_root.append(adult_actuator)

    OUT_XML.write_text(ET.tostring(child_root, encoding="unicode"))
    print(f"wrote {OUT_XML} ({len(adult_sig)} actuators from {ADULT_XML.name}, "
          f"body/geometry from {CHILD_XML.name})")


if __name__ == "__main__":
    main()
