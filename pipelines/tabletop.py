import argparse
from typing import List


def configure_args(args: argparse.Namespace) -> argparse.Namespace:
  out = argparse.Namespace(**vars(args))
  if not str(getattr(out, "judge_model", "")).strip():
    out.judge_model = "deepseek-chat"
  if not str(getattr(out, "control_mode", "")).strip():
    out.control_mode = "pd_ee_delta_pose"
  return out


def control_mode_candidates(args: argparse.Namespace) -> List[str]:
  preferred = str(getattr(args, "control_mode", "") or "pd_ee_delta_pose").strip()
  return [preferred, "pd_joint_delta_pos", "pd_joint_pos"]
