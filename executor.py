import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np

from config import ExecutionConfig
from scene_capture import capture_frame_midexecution


@dataclass
class StepResult:
    step_idx: int
    instruction: str
    success: bool
    message: str = ""
    error: str = ""


@dataclass
class ExecutionResult:
    total_steps: int
    success_steps: int
    success_rate: float
    logs: List[StepResult] = field(default_factory=list)
    replan_count: int = 0


def execute_plan(
    controller: Any,
    plan: List[str],
    env: Optional[gym.Env] = None,
    planner: Optional[Any] = None,
    task: str = "",
    scene_objects: Optional[List[Dict[str, str]]] = None,
    config: Optional[ExecutionConfig] = None,
) -> ExecutionResult:
    """Execute low-level plan with optional visual-feedback replanning.

    If planner and env are provided and config.replan_on_failure is True:
    - After a step failure, capture current frame via capture_frame_midexecution(env)
    - Call planner.replan_from_feedback() with completed/remaining context
    - Replace remaining steps with replanned steps
    - Retry up to config.max_replan_attempts times
    """
    if config is None:
        config = ExecutionConfig()

    if hasattr(controller, "restore_scene"):
        controller.restore_scene()

    steps = [str(x).strip() for x in plan if str(x).strip()]
    logs: List[StepResult] = []
    success_count = 0
    replan_count = 0

    completed: List[str] = []
    remaining = list(steps)

    while remaining:
        step = remaining.pop(0)
        step_idx = len(completed) + 1

        try:
            ret_dict = controller.llm_skill_interact(step)
            success = bool(ret_dict.get("success", False))

            logs.append(
                StepResult(
                    step_idx=step_idx,
                    instruction=step,
                    success=success,
                    message=ret_dict.get("message", ""),
                    error=ret_dict.get("errorMessage", ""),
                )
            )
            print(ret_dict)
            print("-" * 50)

            if success:
                success_count += 1
                completed.append(step)
            else:
                completed.append(step)
                # Attempt replanning on failure
                if (
                    config.replan_on_failure
                    and planner is not None
                    and env is not None
                    and replan_count < config.max_replan_attempts
                    and remaining  # only replan if there are remaining steps
                ):
                    try:
                        print(f"[executor] step failed, attempting replan #{replan_count + 1}...")
                        frame = capture_frame_midexecution(env)
                        failure_info = ret_dict.get("errorMessage", "") or ret_dict.get("message", "")
                        new_remaining = planner.replan_from_feedback(
                            image=frame,
                            task=task,
                            completed_steps=completed,
                            remaining_steps=remaining,
                            failure_info=failure_info,
                            scene_objects=scene_objects,
                        )
                        if new_remaining:
                            remaining = new_remaining
                            replan_count += 1
                            print(f"[executor] replanned remaining steps: {remaining}")
                    except Exception as replan_exc:
                        print(f"[executor] replan failed: {replan_exc}")

        except Exception as exc:
            traceback.print_exc()
            logs.append(
                StepResult(
                    step_idx=step_idx,
                    instruction=step,
                    success=False,
                    message=f"Exception: {exc}",
                    error=str(exc),
                )
            )
            completed.append(step)

    total = len(logs)
    sr = 0.0 if total == 0 else success_count / total
    return ExecutionResult(
        total_steps=total,
        success_steps=success_count,
        success_rate=sr,
        logs=logs,
        replan_count=replan_count,
    )


def execute_plan_simple(controller: Any, plan: List[str]) -> ExecutionResult:
    """Simple execution without replanning (backward compatible)."""
    config = ExecutionConfig(replan_on_failure=False)
    return execute_plan(controller=controller, plan=plan, config=config)
