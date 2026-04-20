from typing import Any

from .humanoid_controller import HumanoidController
from .tabletop_controller import TabletopController


def is_humanoid_scene(scene: str) -> bool:
	s = str(scene or "").casefold().strip()
	return any(tok in s for tok in ["unitree", "humanoid", "h1", "g1"])


def build_controller(env: Any, scene: str, pipeline_name: str = "default") -> Any:
	if is_humanoid_scene(scene) or str(pipeline_name).casefold().strip() == "humanoid":
		return HumanoidController(env)
	return TabletopController(env)


__all__ = [
	"TabletopController",
	"HumanoidController",
	"build_controller",
	"is_humanoid_scene",
]
