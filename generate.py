import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import gymnasium as gym
import mani_skill.envs  # noqa: F401  registers ManiSkill environments

# Register custom environments (PartsSorting, etc.)
import sys as _sys
_parent_dir = str(__import__("pathlib").Path(__file__).resolve().parent.parent)
if _parent_dir not in _sys.path:
    _sys.path.insert(0, _parent_dir)
import imple_new.envs  # noqa: F401
import numpy as np
from tqdm import tqdm

from config import PipelineConfig, load_config
from scene_capture import capture_frame, extract_scene_objects, save_image
from vlm_client import LLMClient

# Known tabletop env IDs from ManiSkill3 (excluding YCB/TwoRobot that need extra assets)
TABLETOP_ENV_IDS = [
    "PickCube-v1",
    "StackCube-v1",
    "PushCube-v1",
    "PullCube-v1",
    "PokeCube-v1",
    "PlaceSphere-v1",
    "RollBall-v1",
    "PushT-v1",
    "LiftPegUpright-v1",
    "PegInsertionSide-v1",
    "PlugCharger-v1",
    "TurnFaucet-v1",
    "PullCubeTool-v1",
    "PartsSorting-v1",
]


def make_tabletop_env(scene_id: str, config: PipelineConfig) -> gym.Env:
    """Create a tabletop ManiSkill3 environment with fallback control modes."""
    control_modes = [config.control.control_mode] + list(config.control.fallback_control_modes)

    kwargs_base = {
        "obs_mode": "rgb",
        "max_episode_steps": config.control.max_episode_steps,
        "render_mode": config.control.render_mode,
        "render_backend": config.control.render_backend,
        "sim_backend": config.control.sim_backend,
    }

    last_error: Optional[Exception] = None
    for mode in control_modes:
        try:
            return gym.make(
                scene_id,
                control_mode=mode,
                sensor_configs=dict(shader_pack=config.control.shader),
                human_render_camera_configs=dict(shader_pack=config.control.shader),
                viewer_camera_configs=dict(shader_pack=config.control.shader),
                **kwargs_base,
            )
        except Exception as exc:
            last_error = exc
            msg = str(exc).casefold()
            if "supported mode" in msg or "not in supported modes" in msg:
                continue
            # Try without sensor/camera configs
            try:
                return gym.make(scene_id, control_mode=mode, **kwargs_base)
            except Exception as exc2:
                last_error = exc2
                continue

    # Final fallback: no control_mode specified
    try:
        return gym.make(scene_id, **kwargs_base)
    except Exception as exc:
        last_error = exc

    raise RuntimeError(f"Unable to create env for {scene_id}: {last_error}")


def generate_task_instruction(
    llm_client: LLMClient,
    model: str,
    image: np.ndarray,
    scene_id: str,
    objects: List[Dict[str, str]],
) -> str:
    """Use LLM to generate a task instruction for the scene."""
    object_names = [obj["name"] for obj in objects if obj.get("name")][:30]

    sys_prompt = (
        "You are generating one candidate robot instruction for downstream policy testing. "
        "The instruction should describe plausible interactions with visible and interactive objects. "
        "Output exactly one sentence in imperative mood."
    )
    user_prompt = (
        f"Scene id: {scene_id}\n"
        f"Known interactive objects: {object_names}\n"
        "Goal: Generate one possible manipulation instruction for this scene.\n"
        "Constraints:\n"
        "1) Include 1-3 concrete actions (pick, move, place, push, pull, open, close, rotate, insert).\n"
        "2) Use object names from the provided list; do not invent objects.\n"
        "3) Keep it concise, physically plausible, and executable by a robot arm.\n"
        "4) Return only the final instruction sentence.\n"
        "Example: Pick up the cube, move it slightly to the right, and place it on the table."
    )

    # Try with image first
    try:
        text = llm_client.chat_with_image(
            model=model,
            system_prompt=sys_prompt,
            user_text=user_prompt,
            image=image,
            temperature=0.4,
            max_tokens=80,
        )
        if text.strip():
            return text.strip()
    except Exception:
        pass

    # Text-only fallback
    text = llm_client.chat(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
        max_tokens=80,
    )
    return text.strip() or "Pick up the cube, move it slightly to the right, and place it on the table."


def generate_dataset(args: argparse.Namespace) -> None:
    """Main dataset generation loop."""
    config = load_config()
    llm_client = LLMClient(config.api)

    out_json = Path(args.output_json)
    image_root = Path(args.image_dir)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)

    if args.scene_ids:
        scene_ids = [x.strip() for x in args.scene_ids.split(",") if x.strip()]
    else:
        scene_ids = list(TABLETOP_ENV_IDS)

    if args.max_scenes > 0:
        scene_ids = scene_ids[: args.max_scenes]

    model = args.model or config.models.generate_model

    print(f"[generate] scenes to process: {scene_ids}")
    records: List[Dict[str, Any]] = []

    for scene_idx, scene_id in tqdm(
        enumerate(scene_ids), total=len(scene_ids), desc="Processing scenes"
    ):
        for sample_idx in range(args.samples_per_scene):
            seed = args.seed + scene_idx * args.samples_per_scene + sample_idx
            env: Optional[gym.Env] = None
            try:
                env = make_tabletop_env(scene_id, config)
                obs, _ = env.reset(seed=seed)
                image = capture_frame(obs, env)
                objects = extract_scene_objects(env)

                scene_dir = image_root / scene_id
                scene_dir.mkdir(parents=True, exist_ok=True)
                image_name = f"seed_{seed:06d}.png"
                image_path = scene_dir / image_name
                save_image(image, str(image_path))

                instruction = generate_task_instruction(
                    llm_client=llm_client,
                    model=model,
                    image=image,
                    scene_id=scene_id,
                    objects=objects,
                )

                record = {
                    "scene_id": scene_id,
                    "task_category": "tabletop",
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
            "source": "ManiSkill3 tabletop tasks",
            "model": model,
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
    parser = argparse.ArgumentParser(description="Generate tabletop benchmark dataset.")
    parser.add_argument(
        "--output-json", type=str, default="imple_new/outputs/tabletop_tasks.json",
    )
    parser.add_argument(
        "--image-dir", type=str, default="imple_new/outputs/generated_frames",
    )
    parser.add_argument(
        "--scene-ids", type=str, default="",
        help="Comma-separated env ids. Empty means use all tabletop envs.",
    )
    parser.add_argument("--max-scenes", type=int, default=0, help="Limit scenes. 0 = all.")
    parser.add_argument("--samples-per-scene", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=str, default="deepseek-chat")
    args = parser.parse_args()
    generate_dataset(args)
