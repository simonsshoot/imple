import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import gymnasium as gym
import mani_skill.envs  # noqa: F401  registers ManiSkill environments

# Register custom environments (PartsSorting, etc.)
import importlib, sys as _sys
_envs_dir = str(Path(__file__).resolve().parent / "envs")
if _envs_dir not in _sys.path:
    _sys.path.insert(0, _envs_dir)
_parent_dir = str(Path(__file__).resolve().parent.parent)
if _parent_dir not in _sys.path:
    _sys.path.insert(0, _parent_dir)
import imple_new.envs  # noqa: F401
import numpy as np
from mani_skill.utils.wrappers.record import RecordEpisode

from config import PipelineConfig, load_config
from controller import TabletopController
from evaluator import (
    calc_grounding_hallucination,
    calc_task_success,
    collect_metrics,
    extract_gt_objects,
    extract_perceived_objects,
    get_expected_refusal,
    is_refusal_text,
    judge_plan,
)
from executor import ExecutionResult, execute_plan
from planner import TabletopPlanner
from scene_capture import capture_frame, extract_scene_objects, save_image
from vlm_client import LLMClient


def _safe_name(text: str) -> str:
    """Make a safe filename from arbitrary text."""
    keep = []
    for ch in str(text):
        if ch.isalnum() or ch in {"-", "_", "."}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "sample"


def create_env(scene: str, config: PipelineConfig) -> gym.Env:
    """Create a ManiSkill tabletop env with control mode fallback."""
    control_modes = [config.control.control_mode] + list(config.control.fallback_control_modes)

    last_error: Optional[Exception] = None
    for mode in control_modes:
        try:
            if config.control.obs_mode == "state":
                env = gym.make(
                    scene,
                    obs_mode="state",
                    control_mode=mode,
                    render_mode=config.control.render_mode,
                    render_backend=config.control.render_backend,
                    sim_backend=config.control.sim_backend,
                )
            else:
                env = gym.make(
                    scene,
                    obs_mode=config.control.obs_mode,
                    control_mode=mode,
                    max_episode_steps=config.control.max_episode_steps,
                    render_mode=config.control.render_mode,
                    render_backend=config.control.render_backend,
                    sensor_configs=dict(shader_pack=config.control.shader),
                    human_render_camera_configs=dict(shader_pack=config.control.shader),
                    viewer_camera_configs=dict(shader_pack=config.control.shader),
                    sim_backend=config.control.sim_backend,
                )
            return env
        except Exception as exc:
            last_error = exc
            msg = str(exc).casefold()
            if "supported mode" in msg or "not in supported modes" in msg:
                print(f"[run] control mode {mode} not supported for {scene}, trying next...")
                continue
            raise

    raise RuntimeError(
        f"Failed to create env for {scene} with control modes {control_modes}: {last_error}"
    )


def run_one(
    config: PipelineConfig,
    llm_client: LLMClient,
    planner: TabletopPlanner,
    scene: str,
    task: str,
    seed: int,
    output_dir: Path,
    run_id: str = "single",
    save_video: bool = False,
    video_fps: int = 20,
) -> Dict[str, Any]:
    """Run a single benchmark item."""
    os.makedirs(output_dir, exist_ok=True)
    env: Optional[gym.Env] = None
    raw_env: Optional[gym.Env] = None

    try:
        raw_env = create_env(scene, config)
        env = raw_env

        if save_video:
            env = RecordEpisode(
                env,
                output_dir=str(output_dir),
                save_trajectory=False,
                save_video=True,
                video_fps=video_fps,
            )

        obs, _ = env.reset(seed=seed)

        # Capture initial frame for VLM
        img = capture_frame(obs, env)
        save_image(img, str(output_dir / f"initial_frame_{_safe_name(run_id)}.png"))

        # Extract scene objects
        objs_all = extract_scene_objects(env)

        # === Planning ===
        llm_plan_error: Optional[str] = None
        ll_plan: List[str] = []
        try:
            scene_desc, hl_plan, ll_plan = planner.plan_full(
                image=img, task=task, scene_objects=objs_all
            )
        except Exception as exc:
            llm_plan_error = str(exc)
            print(f"[run] planner failed: {exc}")

        llm_plan_before_postprocess = list(ll_plan)

        # Postprocess plan for scene-specific fixes
        ll_plan = TabletopPlanner.postprocess_plan(scene=scene, task=task, plan=ll_plan)

        # === Execution ===
        controller = TabletopController(env)
        exec_result = execute_plan(
            controller=controller,
            plan=ll_plan,
            env=env,
            planner=planner,
            task=task,
            scene_objects=objs_all,
            config=config.execution,
        )

        # === Video ===
        saved_video: Optional[str] = None
        if save_video and hasattr(env, "render_images") and len(env.render_images) < 2:
            noop = np.zeros_like(env.action_space.sample(), dtype=np.float32)
            env.step(noop)

        if save_video and hasattr(env, "flush_video"):
            video_name = f"video_{_safe_name(run_id)}"
            env.flush_video(name=video_name)
            saved_video = str(output_dir / f"{video_name}.mp4")
            print(f"[run] saved video to: {saved_video}")

        # === Metrics ===
        gt_objects = extract_gt_objects(objs_all)
        perceived = extract_perceived_objects(llm_plan_before_postprocess)
        gr, hr = calc_grounding_hallucination(perceived, gt_objects)

        plan_judge_result = judge_plan(
            llm_client=llm_client,
            model=config.models.judge_model,
            image=img,
            task=task,
            plan=ll_plan,
            scene=scene,
            scene_objects=objs_all,
        )
        print(f"[run] plan judge result: {plan_judge_result}")
        refusal_detected = bool(plan_judge_result == "Not Safe") or is_refusal_text(
            llm_plan_error or ""
        )

        planning_success = bool(plan_judge_result == "Pass")
        task_success = bool(exec_result.success_rate >= 1.0)

        exec_dict = {
            "num_total_steps": exec_result.total_steps,
            "num_success_steps": exec_result.success_steps,
            "success_rate": exec_result.success_rate,
            "replan_count": exec_result.replan_count,
            "logs": [
                {
                    "step_idx": s.step_idx,
                    "instruction": s.instruction,
                    "success": s.success,
                    "message": s.message,
                    "errorMessage": s.error,
                }
                for s in exec_result.logs
            ],
        }

        result = {
            "scene_id": scene,
            "seed": int(seed),
            "task_instruction": task,
            "low_level_plan": ll_plan,
            "llm_low_level_plan": llm_plan_before_postprocess,
            "llm_plan_error": llm_plan_error,
            "gt_interactable_objects": gt_objects,
            "perceived_objects": perceived,
            "grounding_recall": gr,
            "hallucination_rate": hr,
            "refusal_detected": refusal_detected,
            "plan_judge_result": plan_judge_result,
            "planning_success": planning_success,
            "task_success": task_success,
            "video_path": saved_video,
            "execution": exec_dict,
        }
        return result
    finally:
        if env is not None:
            env.close()
        elif raw_env is not None:
            raw_env.close()


def run_dataset(config: PipelineConfig, args: argparse.Namespace) -> None:
    """Run full dataset in batch mode."""
    dataset_path = Path(args.dataset_json)
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset json not found: {dataset_path}")

    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    data = payload.get("data", []) if isinstance(payload, dict) else []
    if not isinstance(data, list):
        raise ValueError("dataset json format invalid: `data` must be a list")

    start = max(0, int(args.start_index))
    end = len(data)
    if args.max_items > 0:
        end = min(end, start + int(args.max_items))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    llm_client = LLMClient(config.api)
    planner_obj = TabletopPlanner(llm_client, config.models)

    all_results: List[Dict[str, Any]] = []
    for idx in range(start, end):
        item = data[idx]
        scene = str(item.get("scene_id", "")).strip()
        task = str(item.get("task_instruction", "")).strip()
        seed = int(item.get("seed", args.seed))
        if not scene or not task:
            print(f"[run-dataset] skip idx={idx}: missing scene/task")
            continue

        run_id = f"idx{idx:05d}_{scene}_seed{seed:06d}"
        item_dir = output_dir / _safe_name(run_id)
        item_dir.mkdir(parents=True, exist_ok=True)

        print(f"[run-dataset] running idx={idx}, scene={scene}, seed={seed}")
        try:
            result = run_one(
                config=config,
                llm_client=llm_client,
                planner=planner_obj,
                scene=scene,
                task=task,
                seed=seed,
                output_dir=item_dir,
                run_id=run_id,
                save_video=args.save_video,
                video_fps=args.video_fps,
            )
            result["idx"] = idx
            result["expected_refusal"] = get_expected_refusal(item)
            result["status"] = "ok"
            all_results.append(result)
        except Exception as exc:
            print(f"[run-dataset] failed idx={idx}, error={exc}")
            all_results.append(
                {
                    "idx": idx,
                    "scene_id": scene,
                    "seed": seed,
                    "task_instruction": task,
                    "status": "failed",
                    "error": str(exc),
                }
            )

    # Save results
    summary = {
        "dataset_json": str(dataset_path),
        "processed": len(all_results),
        "ok": sum(1 for x in all_results if x.get("status") == "ok"),
        "failed": sum(1 for x in all_results if x.get("status") == "failed"),
        "results": all_results,
    }
    summary_path = output_dir / "run_results.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[run-dataset] saved summary to: {summary_path}")

    # Save metrics
    if args.save_metrics:
        metrics = collect_metrics(all_results)
        metrics_payload = {
            "dataset_json": str(dataset_path),
            "metrics": metrics,
        }
        metrics_path = output_dir / args.metrics_file
        metrics_path.write_text(
            json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[run-dataset] saved metrics to: {metrics_path}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run VLM+ManiSkill3 tabletop benchmark."
    )
    parser.add_argument(
        "--scene", type=str, default="PickCube-v1",
        help="ManiSkill environment ID (single-run mode)",
    )
    parser.add_argument(
        "--task", type=str, default="Pick up the cube and place it on the table",
        help="Task instruction (single-run mode)",
    )
    parser.add_argument("--output-dir", type=str, default="imple_new/outputs")
    parser.add_argument(
        "--dataset-json", type=str, default="",
        help="Path to dataset JSON for batch mode. If set, ignores --scene/--task.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=0, help="0 = all items")
    parser.add_argument("--seed", type=int, default=0)

    # Model arguments
    parser.add_argument("--vlm-model", type=str, default="qwen-vl-max")
    parser.add_argument("--planner-model", type=str, default="deepseek-chat")
    parser.add_argument("--judge-model", type=str, default="deepseek-chat")

    # Execution arguments
    parser.add_argument(
        "--enable-replan", type=str, default="true",
        help="Enable visual-feedback replanning (true/false)",
    )
    parser.add_argument("--max-replan-attempts", type=int, default=3)

    # ManiSkill arguments
    parser.add_argument("--max-episode-steps", type=int, default=240)
    parser.add_argument("--obs-mode", type=str, default="state")
    parser.add_argument("--sim-backend", type=str, default="cpu", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--render-backend", type=str, default="cpu", choices=["cpu", "gpu", "none"])
    parser.add_argument("--render-mode", type=str, default="rgb_array")
    parser.add_argument("--shader", type=str, default="default")

    # Video arguments
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-fps", type=int, default=20)

    # Metrics arguments
    parser.add_argument("--save-metrics", action="store_true")
    parser.add_argument("--metrics-file", type=str, default="metrics_summary.json")

    parser.add_argument("--cpu-only", action="store_true")

    args = parser.parse_args()

    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Build config from args
    config = load_config()
    config.models.vlm_model = args.vlm_model
    config.models.low_level_planner_model = args.planner_model
    config.models.judge_model = args.judge_model
    config.control.obs_mode = args.obs_mode
    config.control.max_episode_steps = args.max_episode_steps
    config.control.render_mode = args.render_mode
    config.control.render_backend = args.render_backend
    config.control.sim_backend = args.sim_backend
    config.control.shader = args.shader
    config.execution.replan_on_failure = args.enable_replan.casefold() in {"true", "1", "yes"}
    config.execution.max_replan_attempts = args.max_replan_attempts

    if args.dataset_json:
        run_dataset(config, args)
    else:
        llm_client = LLMClient(config.api)
        planner_obj = TabletopPlanner(llm_client, config.models)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        result = run_one(
            config=config,
            llm_client=llm_client,
            planner=planner_obj,
            scene=args.scene,
            task=args.task,
            seed=args.seed,
            output_dir=output_dir,
            run_id="single",
            save_video=args.save_video,
            video_fps=args.video_fps,
        )

        if args.save_metrics:
            metrics_path = output_dir / args.metrics_file
            single_metrics = {
                "grounding_recall": result.get("grounding_recall", 0.0),
                "hallucination_rate": result.get("hallucination_rate", 0.0),
                "planning_refusal_rate": 1.0 if result.get("refusal_detected") else 0.0,
                "planning_success_rate": 1.0 if result.get("planning_success") else 0.0,
                "task_success_count": 1 if result.get("task_success") else 0,
                "total_task_count": 1,
                "TSR": 1.0 if result.get("task_success") else 0.0,
            }
            metrics_path.write_text(
                json.dumps(single_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"[run] saved metrics to: {metrics_path}")
        print(f"[run] result: {json.dumps(result, ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    main()
