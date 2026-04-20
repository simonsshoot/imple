import re
from typing import Any, Dict, Optional

import numpy as np

from .tabletop_controller import TabletopController


class HumanoidController(TabletopController):
	"""Humanoid controller entry.

	This class is intentionally separated to keep project architecture clear.
	Current implementation reuses tabletop motion primitives as a baseline,
	while exposing humanoid-specific action names for planner compatibility.
	"""

	def __init__(self, env: Any) -> None:
		super().__init__(env)
		self.actions = list(dict.fromkeys([
			# Preferred short atomic humanoid verbs.
			"stand",
			"stabilize",
			"recover",
			"face_left",
			"face_right",
			"reach_left",
			"reach_right",
			"move_left_to",
			"move_right_to",
			"align",
			"grasp_left",
			"grasp_right",
			"grasp_both",
			"lift",
			"lower",
			"carry",
			"release",
			"retract",
			# Backward-compatible long names kept as aliases.
			"stand_balance",
			"stabilize_torso",
			"recover_balance",
			"face_source_table",
			"turn_to_target_table",
			"align_two_hands",
			"grasp_two_hands",
			"lift_two_hands",
			"carry_box_to_target",
			"lower_box_to_table",
			"release_two_hands",
			"release_right",
			"retract_right",
			"lift_right",
			# Keep existing primitives for backward compatibility
			*self.actions,
		]))

	@staticmethod
	def _result(action: str, ret: str) -> Dict[str, Any]:
		return {
			"action": action,
			"success": len(ret) == 0,
			"message": ret,
			"errorMessage": ret,
		}

	def _get_tcp_pos(self) -> np.ndarray:
		"""Humanoid robots may expose right_tcp / left_tcp instead of tcp."""
		agent = self.u.agent
		for attr in ["right_tcp", "left_tcp", "tcp"]:
			tcp = getattr(agent, attr, None)
			if tcp is not None and hasattr(tcp, "pose") and hasattr(tcp.pose, "p"):
				p = self._to_np(tcp.pose.p)
				if p.ndim == 2:
					p = p[0]
				return p.astype(np.float32)

		# Last fallback: robot base pose.
		if hasattr(agent, "robot") and hasattr(agent.robot, "pose") and hasattr(agent.robot.pose, "p"):
			p = self._to_np(agent.robot.pose.p)
			if p.ndim == 2:
				p = p[0]
			return p.astype(np.float32)

		return np.zeros(3, dtype=np.float32)

	@staticmethod
	def _clean_instruction(text: str) -> str:
		t = str(text).strip().casefold()
		t = re.sub(r"\s+", " ", t)
		return t.strip(" .,;:")

	@staticmethod
	def _strip_prefix(text: str, prefixes) -> str:
		out = text
		for p in prefixes:
			if out.startswith(p):
				return out[len(p):].strip()
		return out.strip()

	def _extract_target(self, text: str) -> Optional[str]:
		if not text:
			return None
		t = self._strip_prefix(text, ["the ", "a ", "an "])
		t = re.sub(r"\b(to|toward|towards|into|onto|on|at)\b", " ", t)
		t = re.sub(r"\s+", " ", t).strip()
		if not t:
			return None
		name, _ = self.extract_number_from_string(t)
		name = str(name).strip()
		return name or None

	def _move_to_target(self, target: str, lateral: float = 0.0, z_offset: float = 0.12) -> str:
		ent = self._resolve_entity(target)
		if ent is None:
			return f"Cannot find {target}"
		pos = self._get_entity_pos(ent).copy()
		pos[1] += lateral
		pos[2] += z_offset
		self._move_to(pos, grip=-1.0 if self.held_object_name else 1.0, steps=30)
		self._hold_position(grip=-1.0 if self.held_object_name else 1.0, steps=4)
		return ""

	def _execute_humanoid_atomic(self, instruction: str, text: str) -> Optional[Dict[str, Any]]:
		parts = text.split(" ", 1)
		verb = parts[0]
		payload = parts[1].strip() if len(parts) > 1 else ""

		# Alias normalization to short atomic names.
		alias = {
			"stand_balance": "stand",
			"stabilize_torso": "stabilize",
			"recover_balance": "recover",
			"face_source_table": "face_left",
			"turn_to_target_table": "face_right",
			"align_two_hands": "align",
			"grasp_two_hands": "grasp_both",
			"lift_two_hands": "lift",
			"carry_box_to_target": "carry",
			"lower_box_to_table": "lower",
			"release_two_hands": "release",
			"release_right": "release",
			"retract_right": "retract",
			"lift_right": "lift",
		}
		verb = alias.get(verb, verb)

		if verb in {"stand", "stabilize", "recover"}:
			self._hold_position(grip=1.0, steps=14)
			return self._result(instruction, "")

		if verb == "face_left":
			return self._result(instruction, self.rotate_left())
		if verb == "face_right":
			return self._result(instruction, self.rotate_right())

		if verb in {"reach_left", "move_left_to"}:
			target = self._extract_target(payload)
			if target is None:
				self.move_left()
				return self._result(instruction, "")
			return self._result(instruction, self._move_to_target(target, lateral=0.08))

		if verb in {"reach_right", "move_right_to"}:
			target = self._extract_target(payload)
			if target is None:
				self.move_right()
				return self._result(instruction, "")
			return self._result(instruction, self._move_to_target(target, lateral=-0.08))

		if verb == "align":
			target = self._extract_target(payload)
			if target is None:
				self.move_left()
				self.move_right()
				self._hold_position(grip=-1.0 if self.held_object_name else 1.0, steps=4)
				return self._result(instruction, "")
			ret = self._move_to_target(target, lateral=0.05)
			if ret:
				return self._result(instruction, ret)
			ret = self._move_to_target(target, lateral=-0.05)
			return self._result(instruction, ret)

		if verb in {"grasp_left", "grasp_right", "grasp_both"}:
			target = self._extract_target(payload)
			if target is None:
				return self._result(instruction, "Nothing Done. Missing grasp target")
			return self._result(instruction, self.pick(target, None))

		if verb == "lift":
			self._move_by([0.0, 0.0, 0.10], steps=20, grip=-1.0 if self.held_object_name else 1.0)
			return self._result(instruction, "")

		if verb == "lower":
			self._move_by([0.0, 0.0, -0.08], steps=20, grip=-1.0 if self.held_object_name else 1.0)
			return self._result(instruction, "")

		if verb == "carry":
			target = self._extract_target(payload)
			if target:
				return self._result(instruction, self._move_to_target(target, lateral=0.0, z_offset=0.16))
			self._move_forward()
			return self._result(instruction, "")

		if verb == "release":
			return self._result(instruction, self.drop())

		if verb == "retract":
			self.move_back()
			self._hold_position(grip=1.0, steps=4)
			return self._result(instruction, "")

		return None

	def _move_forward(self) -> None:
		self._move_by([0.10, 0.0, 0.0], steps=20, grip=-1.0 if self.held_object_name else 1.0)

	def llm_skill_interact(self, instruction: str):
		text = self._clean_instruction(instruction)

		# Try humanoid-specific atomic parser first.
		ret = self._execute_humanoid_atomic(instruction=instruction, text=text)
		if ret is not None:
			return ret

		# Additional natural-language aliases often produced by LLMs.
		if text.startswith("face ") and "left" in text:
			return self._result(instruction, self.rotate_left())
		if text.startswith("face ") and "right" in text:
			return self._result(instruction, self.rotate_right())

		if text.startswith("reach "):
			target = self._extract_target(text.replace("reach ", "", 1))
			if target:
				return self._result(instruction, self.find(target, None))

		if text.startswith("grasp "):
			target = self._extract_target(text.replace("grasp ", "", 1))
			if target:
				return self._result(instruction, self.pick(target, None))

		if text.startswith("release"):
			return self._result(instruction, self.drop())

		if text.startswith("retract"):
			self.move_back()
			return self._result(instruction, "")

		# Fallback to the existing tabletop parser/executor.
		return super().llm_skill_interact(instruction)
