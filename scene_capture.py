from typing import Any, Dict, List

import gymnasium as gym
import numpy as np
from PIL import Image


def normalize_image(img: Any) -> np.ndarray:
    """Normalize ManiSkill image layouts to HxWxC uint8."""
    arr = np.asarray(img)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[-1] != 1:
        arr = np.squeeze(arr, axis=0)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def capture_frame(obs: Any, env: gym.Env) -> np.ndarray:
    """Capture current frame from observation sensor_data or env.render()."""
    # Try sensor_data first
    if isinstance(obs, dict) and "sensor_data" in obs and len(obs["sensor_data"]) > 0:
        cam_name = next(iter(obs["sensor_data"]))
        rgb = obs["sensor_data"][cam_name]["rgb"]
        if hasattr(rgb, "cpu"):
            return normalize_image(rgb[0].cpu().numpy())
        return normalize_image(rgb[0])

    # Fallback to env.render()
    frame = env.render()
    if frame is not None:
        return normalize_image(frame)

    # Last resort: black image
    return np.zeros((224, 224, 3), dtype=np.uint8)


def capture_frame_midexecution(env: gym.Env) -> np.ndarray:
    """Capture frame during execution for visual feedback loop.

    Uses env.render() since we don't have obs in mid-execution context.
    Returns a black image if rendering is unavailable.
    """
    try:
        frame = env.render()
        if frame is not None:
            return normalize_image(frame)
    except Exception:
        pass
    return np.zeros((224, 224, 3), dtype=np.uint8)


def save_image(img: np.ndarray, path: str, fmt: str = "PNG") -> None:
    """Save numpy image to file."""
    arr = normalize_image(img)
    if arr.ndim == 2:
        pass  # Grayscale, PIL handles HxW directly
    elif arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
        pass
    else:
        raise TypeError(f"Unsupported image shape: {arr.shape}")
    Image.fromarray(arr).save(str(path), format=fmt)


def extract_scene_objects(env: gym.Env) -> List[Dict[str, str]]:
    """Extract actors and articulations from the ManiSkill env scene."""
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

    # Deduplicate
    seen = set()
    deduped: List[Dict[str, str]] = []
    for obj in objects:
        key = (obj["name"], obj["type"])
        if key not in seen:
            seen.add(key)
            deduped.append(obj)
    return deduped
