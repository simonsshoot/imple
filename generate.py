import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import mani_skill.envs  # noqa: F401  # Registers ManiSkill environments
import numpy as np
from tqdm import tqdm
from utils import model_selection, ndarray_to_base64, save_pics


DEFAULT_TASK_CATEGORIES = ["humanoid", "mobile_manipulation", "control", "dexterity"]


def _parse_categories(raw: str) -> List[str]:
	if not raw:
		return list(DEFAULT_TASK_CATEGORIES)
	items = [x.strip() for x in raw.split(",") if x.strip()]
	if not items:
		return list(DEFAULT_TASK_CATEGORIES)
	seen = set()
	out: List[str] = []
	for item in items:
		if item in seen:
			continue
		seen.add(item)
		out.append(item)
	return out


def discover_env_ids_by_categories(
	tasks_root_dir: Path,
	categories: Sequence[str],
	include_two_robot: bool = False,
	include_experimental: bool = False,
) -> Tuple[List[str], Dict[str, str]]:
	"""Discover env IDs from ManiSkill task categories by scanning @register_env decorators."""
	if not tasks_root_dir.exists():
		raise FileNotFoundError(f"tasks root directory not found: {tasks_root_dir}")

	pattern = re.compile(r'^\s*@register_env\("([^"]+)"', re.MULTILINE)
	blocked_tokens = {"SO100", "WidowXAI"}
	if not include_two_robot:
		blocked_tokens.add("TwoRobot")
	if not include_experimental:
		blocked_tokens.update({"YCB"})

	discovered: List[str] = []
	category_by_env: Dict[str, str] = {}

	for category in categories:
		category_dir = tasks_root_dir / category
		if not category_dir.exists():
			print(f"[generate] warn: task category directory not found, skip: {category_dir}")
			continue

		for py_file in sorted(category_dir.rglob("*.py")):
			text = py_file.read_text(encoding="utf-8")
			for env_id in pattern.findall(text):
				if any(token in env_id for token in blocked_tokens):
					continue
				if env_id in category_by_env:
					continue
				discovered.append(env_id)
				category_by_env[env_id] = category

	return discovered, category_by_env


def _normalize_image(img: Any) -> np.ndarray:
	arr = np.asarray(img)
	if arr.ndim == 4:
		arr = arr[0]
	if arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[-1] != 1:
		arr = np.squeeze(arr, axis=0)
	if arr.dtype != np.uint8:
		arr = np.clip(arr, 0, 255).astype(np.uint8)
	return arr


def capture_initial_image(obs: Any, env: gym.Env) -> np.ndarray:
	if isinstance(obs, dict) and "sensor_data" in obs and len(obs["sensor_data"]) > 0:
		cam_name = next(iter(obs["sensor_data"]))
		rgb = obs["sensor_data"][cam_name]["rgb"]
		if hasattr(rgb, "cpu"):
			return _normalize_image(rgb[0].cpu().numpy())
		return _normalize_image(rgb[0])

	frame = env.render()
	if frame is None:
		raise RuntimeError("Unable to capture initial frame from observation or render().")
	return _normalize_image(frame)


def extract_scene_objects(env: gym.Env) -> List[Dict[str, str]]:
	scene = env.unwrapped.scene
	objects: List[Dict[str, str]] = []

	actors = getattr(scene, "actors", {})
	if isinstance(actors, dict):
		for name in actors:
			objects.append({"name": str(name), "type": "actor"})
	else:
		for actor in actors:
			name = getattr(actor, "name", "")
			if name:
				objects.append({"name": str(name), "type": "actor"})

	articulations = getattr(scene, "articulations", {})
	if isinstance(articulations, dict):
		for name in articulations:
			objects.append({"name": str(name), "type": "articulation"})
	else:
		for art in articulations:
			name = getattr(art, "name", "")
			if name:
				objects.append({"name": str(name), "type": "articulation"})

	deduped: List[Dict[str, str]] = []
	seen = set()
	for obj in objects:
		key = (obj["name"], obj["type"])
		if key in seen:
			continue
		seen.add(key)
		deduped.append(obj)
	return deduped


def generate_task_instruction(
	image: np.ndarray,
	scene_id: str,
	task_category: str,
	objects: List[Dict[str, str]],
	model: str = "deepseek-chat",
) -> str:
	client, selected_model = model_selection(model)
	object_names = [obj["name"] for obj in objects if obj.get("name")]
	object_names = object_names[:30]

	sys_prompt = (
		"You are generating one candidate robot instruction for downstream policy testing. "
		"The instruction should describe plausible interactions with visible and interactive objects or the embodied agent itself. "
		"Output exactly one sentence in imperative mood."
	)
	user_prompt = (
		f"Scene id: {scene_id}\n"
		f"Task category: {task_category}\n"
		f"Known interactive objects: {object_names}\n"
		"Goal:\n"
		"Generate one possible manipulation instruction that can be used as a test case for this scene.\n"
		"Constraints:\n"
		"1) Include 1-3 concrete interaction actions such as pick, move, place, push, pull, open, close, rotate, insert, or press.\n"
		"2) Prefer object names from the provided interactive object list; do not invent unrelated objects. If object list is empty, generate a physically plausible embodied-control instruction (e.g., stand, walk, run, balance, open cabinet, carry object if available).\n"
		"3) Keep the sentence concise, physically plausible, and executable by a robot manipulator.\n"
		"4) Return only the final instruction sentence with no explanation or extra formatting.\n"
		"One-shot example:\n"
		"Pick up the cube, move it slightly to the right, and place it on the table."
	)

	encoded_image = ndarray_to_base64(image)
	# Try vision input first. If model endpoint does not support images, fallback to text-only.
	try:
		resp = client.chat.completions.create(
			model=selected_model,
			messages=[
				{"role": "system", "content": sys_prompt},
				{
					"role": "user",
					"content": [
						{"type": "text", "text": user_prompt},
						{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}},
					],
				},
			],
			temperature=0.4,
			max_tokens=80,
		)
		text = (resp.choices[0].message.content or "").strip()
		if text:
			return text
	except Exception:
		pass

	resp = client.chat.completions.create(
		model=selected_model,
		messages=[
			{"role": "system", "content": sys_prompt},
			{"role": "user", "content": user_prompt},
		],
		temperature=0.4,
		max_tokens=80,
	)
	text = (resp.choices[0].message.content or "").strip()
	return text or "Pick up the cube, move it slightly to the right, and place it on the table."


def make_env(scene_id: str, args: argparse.Namespace) -> gym.Env:
	kwargs_common = {
		"obs_mode": args.obs_mode,
		"max_episode_steps": args.max_episode_steps,
		"render_mode": args.render_mode,
		"render_backend": args.render_backend,
		"sim_backend": args.sim_backend,
	}

	attempt_kwargs: List[Dict[str, Any]] = [
		{
			**kwargs_common,
			"control_mode": args.control_mode,
			"sensor_configs": dict(shader_pack=args.shader),
			"human_render_camera_configs": dict(shader_pack=args.shader),
			"viewer_camera_configs": dict(shader_pack=args.shader),
		},
		{
			**kwargs_common,
			"sensor_configs": dict(shader_pack=args.shader),
			"human_render_camera_configs": dict(shader_pack=args.shader),
			"viewer_camera_configs": dict(shader_pack=args.shader),
		},
		{**kwargs_common, "control_mode": args.control_mode},
		{**kwargs_common},
	]

	last_error: Optional[Exception] = None
	for idx, kwargs in enumerate(attempt_kwargs, start=1):
		try:
			return gym.make(scene_id, **kwargs)
		except Exception as exc:
			last_error = exc
			print(f"[generate] env make retry {idx}/{len(attempt_kwargs)} for {scene_id}: {exc}")

	raise RuntimeError(f"Unable to create env for {scene_id}: {last_error}")


def run(args: argparse.Namespace) -> None:
	out_json = Path(args.output_json)
	image_root = Path(args.image_dir)
	out_json.parent.mkdir(parents=True, exist_ok=True)
	image_root.mkdir(parents=True, exist_ok=True)

	category_by_env: Dict[str, str] = {}
	if args.scene_ids:
		scene_ids = [x.strip() for x in args.scene_ids.split(",") if x.strip()]
		for scene_id in scene_ids:
			category_by_env[scene_id] = "custom"
	else:
		scene_ids, category_by_env = discover_env_ids_by_categories(
			tasks_root_dir=Path(args.tasks_root_dir),
			categories=_parse_categories(args.task_categories),
			include_two_robot=args.include_two_robot,
			include_experimental=args.include_experimental,
		)

	if args.max_scenes > 0:
		scene_ids = scene_ids[: args.max_scenes]

	print(f"[generate] scenes to process: {scene_ids}")
	records: List[Dict[str, Any]] = []

	for scene_idx, scene_id in tqdm(enumerate(scene_ids), total=len(scene_ids), desc="Processing scenes"):
		for sample_idx in range(args.samples_per_scene):
			seed = args.seed + scene_idx * args.samples_per_scene + sample_idx
			env: Optional[gym.Env] = None
			try:
				env = make_env(scene_id, args)
				obs, _ = env.reset(seed=seed)
				image = capture_initial_image(obs, env)
				objects = extract_scene_objects(env)
				task_category = category_by_env.get(scene_id, "unknown")

				scene_dir = image_root / scene_id
				scene_dir.mkdir(parents=True, exist_ok=True)
				image_name = f"seed_{seed:06d}.png"
				image_path = scene_dir / image_name
				save_pics(image, image_path)

				instruction = generate_task_instruction(
					image=image,
					scene_id=scene_id,
					task_category=task_category,
					objects=objects,
					model=args.model,
				)

				record = {
					"scene_id": scene_id,
					"task_category": task_category,
					"seed": seed,
					"image_path": str(image_path),
					"task_instruction": instruction,
					"objects": objects,
				}
				records.append(record)
				print(f"[generate] ok: {scene_id} seed={seed} -> {instruction}")
			except Exception as exc:
				print(f"[generate] skip: {scene_id} seed={seed}, error={exc}")
			finally:
				if env is not None:
					env.close()

	payload = {
		"meta": {
			"created_at": datetime.utcnow().isoformat() + "Z",
			"source": "ManiSkill tasks",
			"task_categories": _parse_categories(args.task_categories),
			"model": args.model,
			"samples_per_scene": args.samples_per_scene,
			"scene_count": len(scene_ids),
			"record_count": len(records),
		},
		"data": records,
	}

	out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
	print(f"[generate] saved dataset json to: {out_json}")
	print(f"[generate] total records: {len(records)}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Generate ManiSkill scene snapshots and DeepSeek task instructions."
	)
	parser.add_argument(
		"--tasks-root-dir",
		type=str,
		default="../ManiSkill/mani_skill/envs/tasks",
		help="Path to ManiSkill task root directory.",
	)
	parser.add_argument(
		"--task-categories",
		type=str,
		default=",".join(DEFAULT_TASK_CATEGORIES),
		help="Comma-separated task categories under tasks root, e.g. humanoid,mobile_manipulation,control,dexterity.",
	)
	parser.add_argument(
		"--scene-ids",
		type=str,
		default="",
		help="Optional comma-separated env ids. Empty means auto-discover envs from task categories.",
	)
	parser.add_argument("--max-scenes", type=int, default=0, help="Limit number of scenes. 0 means no limit.")
	parser.add_argument("--samples-per-scene", type=int, default=2, help="How many seeds to sample per scene.")
	parser.add_argument("--seed", type=int, default=0, help="Base seed for deterministic generation.")
	parser.add_argument("--model", type=str, default="deepseek-chat", help="LLM model name for instruction generation.")
	parser.add_argument("--output-json", type=str, default="outputs/maniskill_tasks.json", help="Output JSON path.")
	parser.add_argument(
		"--image-dir",
		type=str,
		default="outputs/generated_first_frames",
		help="Directory to save scene initialization images.",
	)
	parser.add_argument("--include-two-robot", action="store_true", help="Include two-robot tabletop tasks.")
	parser.add_argument(
		"--include-experimental",
		action="store_true",
		help="Include tasks that may require extra assets (e.g. YCB).",
	)

	parser.add_argument("--obs-mode", type=str, default="rgb", help="Observation mode.")
	parser.add_argument(
		"--control-mode",
		type=str,
		default="pd_ee_delta_pose",
		help="Preferred control mode for compatible environments.",
	)
	parser.add_argument("--max-episode-steps", type=int, default=120, help="Max episode steps for env creation.")
	parser.add_argument("--sim-backend", type=str, default="cpu", choices=["auto", "cpu", "gpu"])
	parser.add_argument("--render-backend", type=str, default="cpu", choices=["cpu", "gpu", "none"])
	parser.add_argument("--render-mode", type=str, default="rgb_array", choices=["rgb_array", "sensors", "all", "none"])
	parser.add_argument("--shader", type=str, default="default", choices=["default", "rt", "rt-fast"])
	args = parser.parse_args()
	run(args)
