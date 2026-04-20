import argparse
import json
from pathlib import Path
from typing import Dict, List


# Action specs tailored to official ManiSkill humanoid tasks:
# - UnitreeH1Stand-v1 / UnitreeG1Stand-v1
# - UnitreeG1PlaceAppleInBowl-v1
# - UnitreeG1TransportBox-v1
HUMANOID_TASK_ATOMIC_ACTIONS: Dict[str, List[Dict[str, str]]] = {
  "humanoid_stand": [
    {
      "action": "stand_balance()",
      "explanation": "Maintain upright stable posture without falling.",
      "requires_new_controller": "yes",
    },
    {
      "action": "stabilize_torso()",
      "explanation": "Dampen torso sway and keep center of mass over support polygon.",
      "requires_new_controller": "yes",
    },
    {
      "action": "recover_balance()",
      "explanation": "Perform corrective joint adjustments when robot is close to falling.",
      "requires_new_controller": "yes",
    },
  ],
  "humanoid_pick_place": [
    {
      "action": "find(apple)",
      "explanation": "Locate the apple in the scene.",
      "requires_new_controller": "no",
    },
    {
      "action": "reach_right(apple)",
      "explanation": "Move right hand near the apple grasp pose.",
      "requires_new_controller": "yes",
    },
    {
      "action": "grasp_right(apple)",
      "explanation": "Close right hand to stably grasp the apple.",
      "requires_new_controller": "yes",
    },
    {
      "action": "lift_right(apple)",
      "explanation": "Lift grasped apple while maintaining grasp stability.",
      "requires_new_controller": "yes",
    },
    {
      "action": "find(bowl)",
      "explanation": "Locate the bowl goal position.",
      "requires_new_controller": "no",
    },
    {
      "action": "move_right_to(bowl)",
      "explanation": "Move right hand above the bowl before release.",
      "requires_new_controller": "yes",
    },
    {
      "action": "release_right()",
      "explanation": "Open right hand to drop apple into bowl.",
      "requires_new_controller": "yes",
    },
    {
      "action": "retract_right()",
      "explanation": "Retract right hand away from bowl after release.",
      "requires_new_controller": "yes",
    },
  ],
  "humanoid_transport_box": [
    {
      "action": "face_source_table()",
      "explanation": "Rotate body to face table with the box.",
      "requires_new_controller": "yes",
    },
    {
      "action": "find(box)",
      "explanation": "Locate the transport box.",
      "requires_new_controller": "no",
    },
    {
      "action": "align_two_hands(box)",
      "explanation": "Align both hands to left/right grasp points on the box.",
      "requires_new_controller": "yes",
    },
    {
      "action": "grasp_two_hands(box)",
      "explanation": "Establish stable two-hand contact for lifting.",
      "requires_new_controller": "yes",
    },
    {
      "action": "lift_two_hands(box)",
      "explanation": "Lift the box while keeping two-hand grasp.",
      "requires_new_controller": "yes",
    },
    {
      "action": "turn_to_target_table()",
      "explanation": "Rotate body toward destination table.",
      "requires_new_controller": "yes",
    },
    {
      "action": "carry_box_to_target()",
      "explanation": "Transport the box to destination table area.",
      "requires_new_controller": "yes",
    },
    {
      "action": "lower_box_to_table()",
      "explanation": "Lower box to table surface with controlled contact.",
      "requires_new_controller": "yes",
    },
    {
      "action": "release_two_hands()",
      "explanation": "Open both hands and release the box.",
      "requires_new_controller": "yes",
    },
  ],
}


# Subset that is directly compatible with current imple/controller.py verbs.
CONTROLLER_COMPATIBLE_ACTIONS: List[Dict[str, str]] = [
  {
    "action": "find(target_obj: str)",
    "explanation": "Locate target object and move near it.",
  },
  {
    "action": "pick(obj_name: str)",
    "explanation": "Pick one target object with current end-effector policy.",
  },
  {
    "action": "put(receptacle_name: str)",
    "explanation": "Place currently held object onto target receptacle.",
  },
  {
    "action": "move_forward()",
    "explanation": "Move end-effector forward by fixed delta.",
  },
  {
    "action": "move_back()",
    "explanation": "Move end-effector backward by fixed delta.",
  },
  {
    "action": "move_left()",
    "explanation": "Move end-effector left by fixed delta.",
  },
  {
    "action": "move_right()",
    "explanation": "Move end-effector right by fixed delta.",
  },
  {
    "action": "rotate_left()",
    "explanation": "Rotate end-effector yaw left.",
  },
  {
    "action": "rotate_right()",
    "explanation": "Rotate end-effector yaw right.",
  },
  {
    "action": "drop()",
    "explanation": "Release currently held object.",
  },
]


def build_humanoid_atomic_actions_jsonl(include_controller_compatible: bool = True) -> List[Dict[str, str]]:
  rows: List[Dict[str, str]] = []
  action_id = 1

  for task_name, task_actions in HUMANOID_TASK_ATOMIC_ACTIONS.items():
    for rec in task_actions:
      rows.append(
        {
          "action_id": action_id,
          "task": task_name,
          "action": rec["action"],
          "explanation": rec["explanation"],
          "requires_new_controller": rec["requires_new_controller"],
        }
      )
      action_id += 1

  if include_controller_compatible:
    for rec in CONTROLLER_COMPATIBLE_ACTIONS:
      rows.append(
        {
          "action_id": action_id,
          "task": "controller_compatible_subset",
          "action": rec["action"],
          "explanation": rec["explanation"],
          "requires_new_controller": "no",
        }
      )
      action_id += 1

  return rows


def dump_jsonl(output_path: Path, include_controller_compatible: bool = True) -> None:
  rows = build_humanoid_atomic_actions_jsonl(
    include_controller_compatible=include_controller_compatible
  )
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with output_path.open("w", encoding="utf-8") as f:
    for rec in rows:
      f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Export candidate atomic actions for ManiSkill humanoid tasks."
  )
  parser.add_argument(
    "--output",
    type=str,
    default="humanoid_atomic_actions.jsonl",
    help="Output jsonl path.",
  )
  parser.add_argument(
    "--without-controller-compatible",
    action="store_true",
    help="Do not append the subset that is compatible with current controller verbs.",
  )
  args = parser.parse_args()

  dump_jsonl(
    output_path=Path(args.output),
    include_controller_compatible=not args.without_controller_compatible,
  )
  print(f"[humanoid_atomic_actions] saved: {args.output}")
