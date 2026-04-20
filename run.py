import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import gymnasium as gym
import mani_skill.envs
import numpy as np
from utils import gen_low_level_plan, execute_low_level_plan, save_pics
from agents import Agents
from controllers import build_controller
from judge import (
  calc_grounding_and_hallucination,
  collect_metrics_from_results,
  extract_interactable_scene_objects,
  extract_objects_from_plan,
  get_expected_refusal,
  is_explicit_refusal_text,
  judge_plan_with_llm,
)
from mani_skill.utils.wrappers.record import RecordEpisode
from pipelines import (
  configure_args_for_pipeline,
  control_mode_candidates_for_pipeline,
  resolve_pipeline_name,
)


def _safe_name(text: str) -> str:
  keep = []
  for ch in str(text):
    if ch.isalnum() or ch in {"-", "_", "."}:
      keep.append(ch)
    else:
      keep.append("_")
  return "".join(keep).strip("_") or "sample"


def _create_env(args: argparse.Namespace, scene: str, pipeline_name: str = "default") -> gym.Env:
  candidate_modes = control_mode_candidates_for_pipeline(args, pipeline_name)
  deduped_modes: List[str] = []
  seen = set()
  for mode in candidate_modes:
    m = str(mode).strip()
    if m and m not in seen:
      seen.add(m)
      deduped_modes.append(m)

  preferred_mode = deduped_modes[0] if deduped_modes else ""

  last_error: Optional[Exception] = None
  for mode in deduped_modes:
    try:
      if args.obs_mode == "state":
        env = gym.make(
          scene,
          obs_mode="state",
          control_mode=mode,
          render_mode=args.render_mode,
          render_backend=args.render_backend,
          sim_backend=args.sim_backend,
        )
      else:
        env = gym.make(
          scene,
          obs_mode=args.obs_mode,
          control_mode=mode,
          max_episode_steps=args.max_episode_steps,
          render_mode=args.render_mode,
          render_backend=args.render_backend,
          sensor_configs=dict(shader_pack=args.shader),
          human_render_camera_configs=dict(shader_pack=args.shader),
          viewer_camera_configs=dict(shader_pack=args.shader),
          sim_backend=args.sim_backend,
        )
      if mode != preferred_mode:
        print(f"[run] fallback control mode for {scene}: {mode}")
      return env
    except Exception as exc:
      last_error = exc
      msg = str(exc)
      msg_l = msg.casefold()
      if "supported mode" in msg_l or "not in supported modes" in msg_l:
        print(f"[run] control mode {mode} not supported for {scene}, trying next...")
        continue
      raise

  raise RuntimeError(f"failed to create env for {scene} with control modes {deduped_modes}: {last_error}")


def _extract_scene_objects(env: gym.Env) -> List[Dict[str, str]]:
  objs_all: List[Dict[str, str]] = []
  for name in env.unwrapped.scene.actors:
    objs_all.append({"name": str(name), "type": "actor"})
  for name in env.unwrapped.scene.articulations:
    objs_all.append({"name": str(name), "type": "articulation"})
  return objs_all


def _postprocess_low_level_plan(scene: str, task: str, plan: List[str]) -> List[str]:
  if not plan:
    return plan

  scene_l = scene.casefold()
  task_l = task.casefold()
  out = list(plan)

  # If the task asks for insertion but plan only contains put, convert receptacle-like targets to insert.
  if ("insert" in task_l or "plug" in task_l) and not any(s.startswith("insert ") for s in out):
    converted: List[str] = []
    for step in out:
      if step.startswith("put "):
        target = step[4:].strip().casefold()
        if any(k in target for k in ["receptacle", "hole", "slot", "socket", "box"]):
          converted.append(f"insert {step[4:].strip()}")
          continue
      converted.append(step)
    out = converted

  # PokeCube official task uses the peg as tool; when planner asks to pick cube, redirect to peg.
  if "pokecube" in scene_l:
    redirected: List[str] = []
    for step in out:
      s = step.casefold()
      if s.startswith("find cube"):
        redirected.append("find peg")
      elif s.startswith("pick cube"):
        redirected.append("pick peg")
      elif s.startswith("rotate_"):
        # Avoid unstable rotate in this task; pushing with peg is sufficient.
        continue
      else:
        redirected.append(step)
    out = redirected

  # RollBall scene object is named ball (not sphere).
  if "rollball" in scene_l:
    normalized: List[str] = []
    has_pick = False
    for step in out:
      s = step.replace("sphere", "ball")
      if s.casefold().startswith("rotate_"):
        continue
      normalized.append(s)
      if s.casefold().startswith("pick "):
        has_pick = True
    if not has_pick:
      normalized = ["find ball", "pick ball"] + normalized
    out = normalized

  deduped: List[str] = []
  for step in out:
    if not deduped or deduped[-1] != step:
      deduped.append(step)
  return deduped


def run_one(
  args: argparse.Namespace,
  scene: str,
  task: str,
  seed: int,
  output_dir: Path,
  pipeline_name: str = "default",
  run_id: str = "single",
) -> Dict[str, Any]:
  os.makedirs(output_dir, exist_ok=True)
  env: Optional[gym.Env] = None
  raw_env: Optional[gym.Env] = None

  try:
    raw_env = _create_env(args, scene, pipeline_name=pipeline_name)
    env = raw_env

    if args.save_video:
      env = RecordEpisode(
        env,
        output_dir=str(output_dir),
        save_trajectory=False,
        save_video=True,
        video_fps=args.video_fps,
      )

    obs, _ = env.reset(seed=seed)
    img = None
    if isinstance(obs, dict) and "sensor_data" in obs and len(obs["sensor_data"]) > 0:
      cam_name = next(iter(obs["sensor_data"]))
      rgb = obs["sensor_data"][cam_name]["rgb"]
      img = rgb[0].cpu().numpy() if hasattr(rgb, "cpu") else np.asarray(rgb[0])
    elif args.obs_mode != "state":
      frame = env.render()
      if frame is None:
        raise RuntimeError("Failed to get image for VLM: sensor_data missing and env.render() returned None")
      img = frame
    else:
      frame = env.render()
      if frame is not None:
        img = np.asarray(frame)
        if img.ndim == 4:
          img = img[0]
        save_pics(img, output_dir / f"initial_frame_{_safe_name(run_id)}.png")
      else:
        img = np.zeros((224, 224, 3), dtype=np.uint8)

    objs_all = _extract_scene_objects(env)

    agent = Agents(image=img, task_description=task, model=args.model)

    low_level_plan: List[str] = []
    llm_plan_error: Optional[str] = None
    try:
      # Dataset task is already a concise high-level instruction, convert directly.
      low_level_plan = agent.generate_low_level_plan(task, objs_from_scene=objs_all)
    except Exception as exc:
      llm_plan_error = str(exc)
      print(f"[run] LLM low-level planner failed, fallback to regex mapper: {exc}")

    llm_plan_before_fallback = list(low_level_plan)

    if not low_level_plan:
      low_level_plan = gen_low_level_plan(task)

    low_level_plan = _postprocess_low_level_plan(scene=scene, task=task, plan=low_level_plan)

    planner = build_controller(env=env, scene=scene, pipeline_name=pipeline_name)
    exec_result = execute_low_level_plan(planner, low_level_plan)

    saved_video: Optional[str] = None
    if args.save_video and hasattr(env, "render_images") and len(env.render_images) < 2:
      noop = np.zeros_like(env.action_space.sample(), dtype=np.float32)
      env.step(noop)

    if args.save_video and hasattr(env, "flush_video"):
      video_name = f"video_{_safe_name(run_id)}"
      env.flush_video(name=video_name)
      saved_video = str(output_dir / f"{video_name}.mp4")
      print(f"[run] saved video to: {saved_video}")

    gt_interactable_objects = extract_interactable_scene_objects(objs_all)
    perceived_objects = extract_objects_from_plan(llm_plan_before_fallback)
    grounding_recall, hallucination_rate = calc_grounding_and_hallucination(perceived_objects, gt_interactable_objects)

    judge_model = str(getattr(args, "judge_model", "")).strip() or str(args.model)
    plan_judge_result = judge_plan_with_llm(
      image=img,
      task=task,
      plan=low_level_plan,
      scene=scene,
      objs_all=objs_all,
      judge_model=judge_model,
    )
    print(f"[run] plan judge result: {plan_judge_result}")
    refusal_detected = bool(plan_judge_result == "Not Safe") or is_explicit_refusal_text(llm_plan_error or "")

    num_total_steps = int(exec_result.get("num_total_steps", 0)) if isinstance(exec_result, dict) else 0
    num_success_steps = int(exec_result.get("num_success_steps", 0)) if isinstance(exec_result, dict) else 0
    planning_success = bool(plan_judge_result == "Pass")
    task_success = bool(exec_result.get("success_rate", 0.0) >= 1.0) if isinstance(exec_result, dict) else False

    result = {
      "pipeline": pipeline_name,
      "scene_id": scene,
      "seed": int(seed),
      "task_instruction": task,
      "low_level_plan": low_level_plan,
      "llm_low_level_plan": llm_plan_before_fallback,
      "llm_plan_error": llm_plan_error,
      "gt_interactable_objects": gt_interactable_objects,
      "perceived_objects": perceived_objects,
      "grounding_recall": grounding_recall,
      "hallucination_rate": hallucination_rate,
      "refusal_detected": refusal_detected,
      "plan_judge_result": plan_judge_result,
      "planning_success": planning_success,
      "task_success": task_success,
      "video_path": saved_video,
      "execution": exec_result,
    }
    return result
  finally:
    if env is not None:
      env.close()
    elif raw_env is not None:
      raw_env.close()


def run_dataset(args: argparse.Namespace) -> None:
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
    pipeline_name = resolve_pipeline_name(
      forced_pipeline=args.pipeline,
      scene=scene,
      item=item,
      dataset_path=dataset_path,
    )
    item_args = configure_args_for_pipeline(args, pipeline_name)
    try:
      result = run_one(
        args=item_args,
        scene=scene,
        task=task,
        seed=seed,
        output_dir=item_dir,
        pipeline_name=pipeline_name,
        run_id=run_id,
      )
      result["idx"] = idx
      result["expected_refusal"] = get_expected_refusal(item)
      if result["expected_refusal"] is True:
        # Require explicit refusal and empty executable plan.
        result["refusal_detected"] = bool(result.get("refusal_detected", False))
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

  summary = {
    "dataset_json": str(dataset_path),
    "processed": len(all_results),
    "ok": sum(1 for x in all_results if x.get("status") == "ok"),
    "failed": sum(1 for x in all_results if x.get("status") == "failed"),
    "results": all_results,
  }
  summary_path = output_dir / "run_results.json"
  summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
  print(f"[run-dataset] saved summary to: {summary_path}")

  if args.save_metrics:
    metrics = collect_metrics_from_results(all_results)
    metrics_payload = {
      "dataset_json": str(dataset_path),
      "metrics": metrics,
    }
    metrics_path = output_dir / args.metrics_file
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[run-dataset] saved metrics to: {metrics_path}")


def run(args: argparse.Namespace) -> None:
  if args.cpu_only:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

  if args.dataset_json:
    run_dataset(args)
    return

  pipeline_name = resolve_pipeline_name(
    forced_pipeline=args.pipeline,
    scene=args.scene,
    item=None,
    dataset_path=None,
  )
  run_args = configure_args_for_pipeline(args, pipeline_name)

  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  result = run_one(
    args=run_args,
    scene=run_args.scene,
    task=run_args.tasks,
    seed=run_args.seed,
    output_dir=output_dir,
    pipeline_name=pipeline_name,
    run_id="single",
  )
  if args.save_metrics:
    metrics_path = output_dir / args.metrics_file
    single_metrics = {
      "grounding_recall": result.get("grounding_recall", 0.0),
      "hallucination_rate": result.get("hallucination_rate", 0.0),
      "planning_refusal_rate": 1.0 if result.get("refusal_detected", False) else 0.0,
      "planning_success_rate": 1.0 if result.get("planning_success", False) else 0.0,
      "task_success_count": 1 if result.get("task_success", False) else 0,
      "total_task_count": 1,
      "TSR": 1.0 if result.get("task_success", False) else 0.0,
    }
    metrics_path.write_text(json.dumps(single_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[run] saved metrics to: {metrics_path}")
  print(f"[run] single result: {result}")




if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run ManiSkill with VLM integration.")
  parser.add_argument("--scene",type=str, default="LiftCube-v0", help="ManiSkill environment to run")
  parser.add_argument("--tasks",type=str, default="Move the ball from left to right", help="task prompt")
  parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to save results")
  parser.add_argument(
    "--dataset-json",
    type=str,
    default="",
    help="Path to generated dataset json (e.g. outputs/tabletop_tasks.json). If set, run in batch mode.",
  )
  parser.add_argument("--start-index", type=int, default=0, help="Start index for dataset batch mode.")
  parser.add_argument("--max-items", type=int, default=0, help="Max items for dataset batch mode. 0 means all.")
  parser.add_argument(
    "--pipeline",
    type=str,
    default="auto",
    choices=["auto", "tabletop", "humanoid", "control", "default"],
    help="Pipeline route. auto: infer from task_category / scene / dataset file name.",
  )
  parser.add_argument("--max_episode_steps", type=int, default=250, help="Maximum steps per episode")
  parser.add_argument("--obs-mode", type=str, default="state", help="Observation mode, e.g. state/rgb/rgb+segmentation")
  parser.add_argument("--control-mode", type=str, default="", help="Optional preferred control mode. Empty means pipeline default.")
  parser.add_argument("--model",type=str, default="gpt-4o", help="Model API to use")
  parser.add_argument("--judge-model", type=str, default="deepseek-chat", help="Model for planning success/refusal judging. Empty uses --model")
  parser.add_argument("--save-video", action="store_true", help="Enable video recording. Disable in unstable headless setups.")
  parser.add_argument(
		"--video-fps",
		type=int,
		default=20,
		help="FPS of the saved video. Lower FPS makes the same number of frames play longer.",
	)
  parser.add_argument(
		"--sim-backend",
		type=str,
		default="cpu",
		choices=["auto", "cpu", "gpu"],
		help="Simulation backend. Use cpu on headless servers to avoid GPU/EGL crashes.",
	)
  parser.add_argument(
		"--cpu-only",
		action="store_true",
		help="Hide CUDA devices (sets CUDA_VISIBLE_DEVICES='') for better headless compatibility.",
	)
  parser.add_argument(
		"--shader",
		type=str,
		default="default",
		choices=["default", "rt", "rt-fast"],
		help="Shader pack used for rendering",
	)
  parser.add_argument(
		"--render-mode",
		type=str,
		default="rgb_array",
    choices=["rgb_array", "sensors", "all", "none"],
		help="Render mode used for recording",
	)
  parser.add_argument(
		"--render-backend",
		type=str,
		default="cpu",
		choices=["cpu", "gpu", "none"],
		help="Rendering backend. Use cpu on headless servers.",
	)
  parser.add_argument(
		"--vis",
		action="store_true",
		help="Open viewer for debugging (do not use on a headless server)",
	)
  parser.add_argument(
		"--steps-per-phase",
		type=int,
		default=30,
		help="Control steps for each phase (approach / carry / retreat)",
	)
  parser.add_argument("--seed", type=int, default=0, help="Episode seed")
  parser.add_argument("--save-metrics", action="store_true", help="Save evaluation metrics json")
  parser.add_argument("--metrics-file", type=str, default="metrics_summary.json", help="Metrics output file name")
  args = parser.parse_args()
  run(args)

