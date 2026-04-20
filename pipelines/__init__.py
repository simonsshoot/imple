import argparse
from pathlib import Path
from typing import Any, Dict, Optional

from . import control, default, humanoid, tabletop


def resolve_pipeline_name(
  forced_pipeline: str,
  scene: str,
  item: Optional[Dict[str, Any]] = None,
  dataset_path: Optional[Path] = None,
) -> str:
  forced = (forced_pipeline or "auto").strip().casefold()
  if forced and forced != "auto":
    if forced == "default":
      return "default"
    return forced

  category = ""
  if isinstance(item, dict):
    category = str(item.get("task_category", "")).strip().casefold()

  scene_l = str(scene or "").strip().casefold()
  dataset_name = str(dataset_path.name).casefold() if dataset_path else ""

  if category in {"humanoid"}:
    return "humanoid"
  if category in {"control"}:
    return "control"
  if category in {"tabletop", "dexterity", "mobile_manipulation"}:
    return "tabletop"

  if "unitree" in scene_l or "humanoid" in scene_l:
    return "humanoid"

  if scene_l.startswith("ms-"):
    return "control"

  if any(x in dataset_name for x in ["humanoid"]):
    return "humanoid"
  if any(x in dataset_name for x in ["control"]):
    return "control"
  if any(x in dataset_name for x in ["tabletop", "dexterity", "mobile_manipulation"]):
    return "tabletop"

  return "default"


def configure_args_for_pipeline(args: argparse.Namespace, pipeline_name: str) -> argparse.Namespace:
  p = (pipeline_name or "default").casefold().strip()
  if p == "control":
    return control.configure_args(args)
  if p == "humanoid":
    return humanoid.configure_args(args)
  if p == "tabletop":
    return tabletop.configure_args(args)
  return default.configure_args(args)


def control_mode_candidates_for_pipeline(args: argparse.Namespace, pipeline_name: str) -> list[str]:
  p = (pipeline_name or "default").casefold().strip()
  if p == "control":
    return control.control_mode_candidates(args)
  if p == "humanoid":
    return humanoid.control_mode_candidates(args)
  if p == "tabletop":
    return tabletop.control_mode_candidates(args)
  return default.control_mode_candidates(args)
