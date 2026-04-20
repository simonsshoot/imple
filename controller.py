from typing import Any

from controllers import HumanoidController, TabletopController, build_controller


# Backward compatibility: existing imports expect Controller to exist.
# Keep default as tabletop; new code should use build_controller for scene-aware routing.
class Controller(TabletopController):
	def __init__(self, env: Any) -> None:
		super().__init__(env)


__all__ = [
	"Controller",
	"TabletopController",
	"HumanoidController",
	"build_controller",
]
