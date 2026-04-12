import argparse
import importlib
import json
import os
from pathlib import Path
import types
import typing
from typing import Any, Dict, Optional, Tuple
import gymnasium as gym
import numpy as np
from PIL import Image


def _patch_pkg_resources_compat() -> None:
  """Provide minimal pkg_resources.resource_filename for legacy SAPIEN imports."""
  try:
    import pkg_resources  # noqa: F401
    return
  except Exception:
    pass

  shim = types.ModuleType("pkg_resources")

  def resource_filename(package_name: str, resource_name: str) -> str:
    pkg = importlib.import_module(package_name)
    pkg_dir = Path(getattr(pkg, "__file__", "")).resolve().parent
    return str(pkg_dir / resource_name)

  shim.resource_filename = resource_filename  # type: ignore[attr-defined]
  os.sys.modules["pkg_resources"] = shim


def _patch_typing_extensions_compat() -> None:
  """Backfill legacy names expected by newer openai on old typing_extensions."""
  try:
    import typing_extensions as te
  except Exception:
    return

  fallbacks = {
    "List": typing.List,
    "Sequence": typing.Sequence,
  }
  for name, value in fallbacks.items():
    if not hasattr(te, name):
      setattr(te, name, value)


_patch_pkg_resources_compat()
_patch_typing_extensions_compat()
from utils import gen_low_level_plan, execute_low_level_plan, save_pics
from agents import Agents
from controller import Controller


def _load_meta_item(meta_json_path: str, item_index: int) -> Tuple[str | None, str | None, Dict[str, Any], int | None]:
  """Load a single scenario entry from generate.py meta.json output."""
  with open(meta_json_path, "r", encoding="utf-8") as f:
    meta = json.load(f)

  items = meta.get("items", [])
  if not isinstance(items, list) or not items:
    raise ValueError(f"Invalid meta json (empty or missing 'items'): {meta_json_path}")
  if item_index < 0 or item_index >= len(items):
    raise IndexError(f"meta index out of range: {item_index}, total={len(items)}")

  item = items[item_index]
  if not isinstance(item, dict):
    raise ValueError(f"Invalid item format at index {item_index}")

  scene = item.get("task_name") or meta.get("task_name")
  instruction = item.get("instruction")
  scenario = item.get("scenario", {})
  seed = item.get("seed", meta.get("seed"))
  if scenario is None:
    scenario = {}
  if not isinstance(scenario, dict):
    raise ValueError(f"Invalid scenario format at index {item_index}: expected dict")
  if seed is not None:
    seed = int(seed)
  return scene, instruction, scenario, seed


def _load_meta_count(meta_json_path: str) -> int:
  with open(meta_json_path, "r", encoding="utf-8") as f:
    meta = json.load(f)
  items = meta.get("items", [])
  if isinstance(items, list):
    return len(items)
  return 0


def _make_env(scene: str, args: argparse.Namespace):
  """Create env from either simpler_env task name or ManiSkill gym id."""
  if scene.startswith("google_robot_") or scene.startswith("widowx_"):
    simpler_env = importlib.import_module("simpler_env")
    return simpler_env.make(scene)

  # Import mani_skill lazily so simpler_env runs do not depend on sapien.physx.
  importlib.import_module("mani_skill.envs")

  if args.obs_mode == "state":
    return gym.make(
      scene,
      obs_mode="state",
      control_mode="pd_ee_delta_pose",
      render_mode=args.render_mode,
      render_backend=args.render_backend,
      sim_backend=args.sim_backend,
    )

  return gym.make(
    scene,
    obs_mode=args.obs_mode,
    control_mode="pd_ee_delta_pose",
    max_episode_steps=args.max_episode_steps,
    render_mode=args.render_mode,
    render_backend=args.render_backend,
    sensor_configs=dict(shader_pack=args.shader),
    human_render_camera_configs=dict(shader_pack=args.shader),
    viewer_camera_configs=dict(shader_pack=args.shader),
    sim_backend=args.sim_backend,
  )


def _extract_image(env, obs: Any, obs_mode: str) -> np.ndarray:
  """Extract an RGB frame from env observation/render outputs."""
  if isinstance(obs, dict) and "sensor_data" in obs and len(obs["sensor_data"]) > 0:
    cam_name = next(iter(obs["sensor_data"]))
    return obs["sensor_data"][cam_name]["rgb"][0].cpu().numpy()

  # simpler_env returns ManiSkill2 observation dict; use the official helper.
  try:
    obs_utils = importlib.import_module("simpler_env.utils.env.observation_utils")
    get_image_from_maniskill2_obs_dict = getattr(obs_utils, "get_image_from_maniskill2_obs_dict")

    image = get_image_from_maniskill2_obs_dict(env, obs)
    if image is not None:
      return image
  except Exception:
    pass

  frame = env.render() if obs_mode != "state" else env.render()
  if frame is None:
    return np.zeros((224, 224, 3), dtype=np.uint8)
  img = np.asarray(frame)
  if img.ndim == 4:
    img = img[0]
  return img


def _capture_frame(env: Any, obs: Any, obs_mode: str) -> np.ndarray:
  """Capture a frame for recording: prefer render(), fallback to observation."""
  try:
    frame = env.render()
    if frame is not None:
      img = np.asarray(frame)
      if img.ndim == 4:
        img = img[0]
      if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]
      if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
      return img.astype(np.uint8)
  except Exception:
    pass

  img = _extract_image(env, obs, obs_mode)
  img = np.asarray(img)
  if img.ndim == 4:
    img = img[0]
  if img.ndim == 3 and img.shape[-1] == 4:
    img = img[..., :3]
  if img.ndim == 2:
    img = np.stack([img, img, img], axis=-1)
  return img.astype(np.uint8)


def _collect_scene_objects(env: Any) -> list[dict[str, str]]:
  """Collect actor/articulation names from env scene in a backend-compatible way."""
  objs_all: list[dict[str, str]] = []
  unwrapped = getattr(env, "unwrapped", env)
  scene_obj = getattr(unwrapped, "scene", None)
  if scene_obj is None:
    scene_obj = getattr(unwrapped, "_scene", None)
  if scene_obj is None:
    print("[run] warning: environment has neither 'scene' nor '_scene'; skip scene object extraction")
    scene_obj = None

  def _entity_name(ent: Any, default_prefix: str, idx: int) -> str:
    name = getattr(ent, "name", None)
    if isinstance(name, str) and name.strip():
      return name
    getter = getattr(ent, "get_name", None)
    if callable(getter):
      try:
        v = getter()
        if isinstance(v, str) and v.strip():
          return v
      except Exception:
        pass
    return f"{default_prefix}_{idx}"

  if scene_obj is not None:
    actors = getattr(scene_obj, "actors", {})
    if isinstance(actors, dict):
      for name in actors.keys():
        objs_all.append({"name": str(name), "type": "actor"})
    else:
      get_all_actors = getattr(scene_obj, "get_all_actors", None)
      if callable(get_all_actors):
        try:
          for i, actor in enumerate(list(get_all_actors())):
            objs_all.append({"name": _entity_name(actor, "actor", i), "type": "actor"})
        except Exception:
          pass

    articulations = getattr(scene_obj, "articulations", {})
    if isinstance(articulations, dict):
      for name in articulations.keys():
        objs_all.append({"name": str(name), "type": "articulation"})
    else:
      get_all_articulations = getattr(scene_obj, "get_all_articulations", None)
      if callable(get_all_articulations):
        try:
          for i, art in enumerate(list(get_all_articulations())):
            objs_all.append({"name": _entity_name(art, "articulation", i), "type": "articulation"})
        except Exception:
          pass

  # Fallback for ManiSkill2_real2sim custom scenes where key actors are exposed
  # on env fields (e.g. episode_objs, sink, episode_source_obj, episode_target_obj).
  seen = {(o.get("type"), o.get("name")) for o in objs_all}

  def _add_obj(name: str, typ: str) -> None:
    key = (typ, name)
    if not name or key in seen:
      return
    seen.add(key)
    objs_all.append({"name": name, "type": typ})

  for ent in getattr(unwrapped, "episode_objs", []) or []:
    _add_obj(_entity_name(ent, "episode_obj", len(objs_all)), "actor")

  for attr_name, typ in [
    ("episode_source_obj", "actor"),
    ("episode_target_obj", "actor"),
    ("sink", "actor"),
  ]:
    ent = getattr(unwrapped, attr_name, None)
    if ent is not None:
      _add_obj(_entity_name(ent, attr_name, len(objs_all)), typ)

  agent = getattr(unwrapped, "agent", None)
  robot = getattr(agent, "robot", None) if agent is not None else None
  if robot is not None:
    name = getattr(robot, "name", None)
    if not isinstance(name, str) or not name.strip():
      name = "robot"
    _add_obj(str(name), "articulation")

  print(f"[run] objs_all count: {len(objs_all)}")
  for i, obj in enumerate(objs_all):
    print(f"[run][obj {i:03d}] type={obj.get('type', '')} name={obj.get('name', '')}")

  return objs_all


def _save_video_from_frames(frames: list[np.ndarray], out_path: Path, fps: int, min_seconds: float = 3.0) -> Optional[Path]:
  """Save frames to mp4; fallback to gif if mp4 backend is unavailable."""
  if len(frames) == 0:
    return None

  arrs: list[np.ndarray] = []
  for f in frames:
    img = np.asarray(f)
    if img.ndim == 4:
      img = img[0]
    if img.ndim == 2:
      img = np.stack([img, img, img], axis=-1)
    if img.ndim == 3 and img.shape[-1] == 4:
      img = img[..., :3]
    arrs.append(img.astype(np.uint8))

  min_frames = max(1, int(max(0.0, float(min_seconds)) * max(1, int(fps))))
  if len(arrs) < min_frames:
    pad = arrs[-1]
    arrs.extend([pad] * (min_frames - len(arrs)))

  try:
    import imageio.v2 as imageio

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(out_path), fps=max(1, int(fps))) as writer:
      for img in arrs:
        writer.append_data(img)
    return out_path
  except Exception:
    gif_path = out_path.with_suffix(".gif")
    pil_frames = [Image.fromarray(x) for x in arrs]
    pil_frames[0].save(
      gif_path,
      save_all=True,
      append_images=pil_frames[1:],
      duration=max(20, int(1000 / max(1, int(fps)))),
      loop=0,
    )
    return gif_path


def _run_single(args: argparse.Namespace, meta_index_override: Optional[int] = None, output_dir_override: Optional[Path] = None) -> None:
  task = args.tasks
  scene = args.scene
  reset_options = None
  reset_seed = args.seed
  active_meta_index = args.meta_index if meta_index_override is None else int(meta_index_override)

  if args.meta_json:
    meta_scene, meta_instruction, meta_options, meta_seed = _load_meta_item(args.meta_json, active_meta_index)
    if meta_scene:
      scene = meta_scene
    if meta_instruction:
      task = meta_instruction
    reset_options = meta_options
    if args.seed is None and meta_seed is not None:
      reset_seed = meta_seed

    print(f"[run] loaded meta item #{active_meta_index} from {args.meta_json}")
    print(f"[run] scene: {scene}")
    print(f"[run] instruction: {task}")

  output_dir = output_dir_override or Path(args.output_dir)
  os.makedirs(output_dir, exist_ok=True)

  if args.cpu_only:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

  env = _make_env(scene, args)
  is_simpler_scene = scene.startswith("google_robot_") or scene.startswith("widowx_")

  if args.save_video:
    if not is_simpler_scene:
      record_mod = importlib.import_module("mani_skill.utils.wrappers.record")
      RecordEpisode = getattr(record_mod, "RecordEpisode")
      env = RecordEpisode(
        env,
        output_dir=str(output_dir),
        save_trajectory=False,
        save_video=True,
        video_fps=args.video_fps,
      )
    else:
      print("[run] save-video enabled for simpler_env: using frame recorder")

  reset_kwargs: Dict[str, Any] = {"seed": reset_seed}
  if reset_options:
    reset_kwargs["options"] = reset_options
  obs, info = env.reset(**reset_kwargs)

  img = _extract_image(env, obs, args.obs_mode)
  save_pics(img, output_dir / "initial_frame.png")

  recorded_frames: list[np.ndarray] = []
  if args.save_video and is_simpler_scene:
    recorded_frames.append(_capture_frame(env, obs, args.obs_mode))

  objs_all = _collect_scene_objects(env)

  agent = Agents(image=img, task_description=task, model=args.model)
  _, plan = agent.multi_agent_vision_planning(objs_all)

  low_level_plan = []
  try:
    low_level_plan = agent.generate_low_level_plan(plan, objs_from_scene=objs_all)
  except Exception as exc:
    print(f"[run] LLM low-level planner failed, fallback to regex mapper: {exc}")

  if not low_level_plan:
    low_level_plan = gen_low_level_plan(plan)

  planner = Controller(env)

  if args.save_video and is_simpler_scene:
    def _on_sim_step(last_obs: Any) -> None:
      recorded_frames.append(_capture_frame(env, last_obs, args.obs_mode))

    planner.frame_callback = _on_sim_step

  def _after_step(_: int, __: str, ___: Dict[str, Any]) -> None:
    if not (args.save_video and is_simpler_scene):
      return
    obs_for_frame = getattr(planner, "last_obs", None)
    frame = _extract_image(env, obs_for_frame, args.obs_mode)
    recorded_frames.append(np.asarray(frame).copy())

  execute_low_level_plan(planner, low_level_plan, step_callback=_after_step)

  if args.save_video and is_simpler_scene:
    # quick diagnostic: if almost all frames are identical, execution likely did not move visually.
    changed = 0
    for i in range(1, len(recorded_frames)):
      if recorded_frames[i].shape == recorded_frames[i - 1].shape and np.any(recorded_frames[i] != recorded_frames[i - 1]):
        changed += 1
    print(f"[run] recorded frames: total={len(recorded_frames)}, changed_pairs={changed}")

    video_path = _save_video_from_frames(
      recorded_frames,
      output_dir / "final_video.mp4",
      fps=args.video_fps,
      min_seconds=float(args.min_video_seconds),
    )
    if video_path is not None:
      print(f"[run] saved video to: {video_path}")

  if args.save_video and (not is_simpler_scene):
    render_images = getattr(env, "render_images", None)
    if render_images is not None and len(render_images) < 2:
      noop = np.zeros_like(env.action_space.sample(), dtype=np.float32)
      env.step(noop)

    flush_video = getattr(env, "flush_video", None)
    if callable(flush_video):
      flush_video(name="final_video")
      print(f"[run] saved video to: {output_dir / 'final_video.mp4'}")

  env.close()

def run(args: argparse.Namespace) -> None:
  if args.meta_json and args.meta_count > 1:
    total = _load_meta_count(args.meta_json)
    if total <= 0:
      raise ValueError(f"Invalid meta file: no items found in {args.meta_json}")

    start = max(0, int(args.meta_index))
    end = min(total, start + int(args.meta_count))
    if start >= end:
      raise ValueError(f"No items to run: start={start}, end={end}, total={total}")

    base_out = Path(args.output_dir)
    for idx in range(start, end):
      case_out = base_out / f"meta_case_{idx}"
      print(f"\n[run] ===== meta case {idx} -> {case_out} =====")
      _run_single(args, meta_index_override=idx, output_dir_override=case_out)
    return

  _run_single(args)




if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run ManiSkill with VLM integration.")
  parser.add_argument("--scene",type=str, default="LiftCube-v0", help="ManiSkill environment to run")
  parser.add_argument("--tasks",type=str, default="Move the ball from left to right", help="task prompt")
  parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to save results")
  parser.add_argument("--max_episode_steps", type=int, default=250, help="Maximum steps per episode")
  parser.add_argument("--obs-mode", type=str, default="state", help="Observation mode, e.g. state/rgb/rgb+segmentation")
  parser.add_argument("--model",type=str, default="gpt-4o", help="Model API to use")
  parser.add_argument("--save-video", action="store_true", help="Enable video recording. Disable in unstable headless setups.")
  parser.add_argument(
		"--video-fps",
		type=int,
		default=20,
		help="FPS of the saved video. Lower FPS makes the same number of frames play longer.",
	)
  parser.add_argument("--min-video-seconds", type=float, default=3.0, help="Minimum duration for exported video in seconds")
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
  parser.add_argument("--meta-json", type=str, default=None, help="Path to generate.py output meta.json")
  parser.add_argument("--meta-index", type=int, default=0, help="Item index in meta.json['items']")
  parser.add_argument("--meta-count", type=int, default=1, help="How many consecutive meta items to run from --meta-index")
  parser.add_argument("--seed", type=int, default=None, help="Episode seed; if omitted and --meta-json is used, fallback to meta seed")
  args = parser.parse_args()
  run(args)

