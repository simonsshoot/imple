import logging
import math
import re
import string
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from mani_skill.utils.structs import Pose

log = logging.getLogger(__name__)


@dataclass
class EntityRef:
	name: str
	entity_type: str  # actor | articulation
	entity: Any


class TabletopController:
	def __init__(self, env: Any) -> None:
		self.env = env
		self.u = env.unwrapped
		self.actions = [
			"find",
			"pick",
			"put",
			"insert",
			"move_left",
			"move_right",
			"move_forward",
			"move_back",
			"rotate_left",
			"rotate_right",
			"rotate_up",
			"rotate_down",
			"open",
			"close",
			"slice",
			"turn on",
			"turn off",
			"drop",
			"throw",
			"break",
			"cook",
			"dirty",
			"clean",
			"fillLiquid",
			"emptyLiquid",
			"pour",
		]
		self.last_obs = None
		self.last_reward = None
		self.last_terminated = False
		self.last_truncated = False
		self.last_info = None
		self.multi_objs_dict: Dict[str, Dict[str, int]] = {}
		self.object_states: Dict[str, Dict[str, Any]] = {}
		self.held_object_name: Optional[str] = None
		self.held_object_soft_attached: bool = False
		self.held_object_offset: Optional[np.ndarray] = None
		self._scene_entities: Dict[str, EntityRef] = {}
		self.restore_scene()

	def restore_scene(self) -> None:
		self._refresh_scene_entities()
		self.multi_objs_dict = {}

	def _refresh_scene_entities(self) -> None:
		self._scene_entities = {}
		scene = self.u.scene
		for name, actor in scene.actors.items():
			self._scene_entities[name] = EntityRef(name=name, entity_type="actor", entity=actor)
		for name, art in scene.articulations.items():
			self._scene_entities[name] = EntityRef(name=name, entity_type="articulation", entity=art)

	@staticmethod
	def natural_word_to_name(w: str) -> str:
		if w == "CD":
			return w
		return "".join([string.capwords(x) for x in w.split()])

	@staticmethod
	def extract_number_from_string(s: str) -> Tuple[str, Optional[int]]:
		match = re.match(r"^(.*\D)\s*(\d+)?$", s)
		if match:
			text_part = match.group(1).strip()
			number_part = int(match.group(2)) if match.group(2) else None
			return text_part, number_part
		return s, None

	@staticmethod
	def split_string_for_fill(s: str) -> Tuple[str, str]:
		parts = s.split()
		if len(parts) < 2:
			return s, "water"
		part1 = " ".join(parts[:-1])
		part2 = parts[-1]
		return part1, part2

	@staticmethod
	def _normalize_name(text: str) -> str:
		return re.sub(r"[^a-z0-9]", "", text.casefold())

	@staticmethod
	def _canonical_object_name(obj_name: str) -> str:
		name = obj_name.strip().casefold()
		synonyms = {
			"orb": "sphere",
			"cup": "mug",
			"goal": "goal_region",
			"goal zone": "goal_region",
			"goal area": "goal_region",
			"goal region": "goal_region",
		}
		return synonyms.get(name, obj_name)

	def _resolve_entity(self, obj_name: str, obj_num: Optional[int] = None) -> Optional[EntityRef]:
		"""实体解析，将自然语言描述的对象名称映射到场景中的实际物理实体"""
		del obj_num  # Not used in current ManiSkill scenes.
		if not self._scene_entities:
			self._refresh_scene_entities()

		obj_name = self._canonical_object_name(obj_name)

		q_norm = self._normalize_name(obj_name)
		if not q_norm:
			return None

		# Explicitly handle common target-region aliases used by LLMs.
		if q_norm in {"goal", "goalregion", "goalzone", "goalarea", "region", "target", "targetregion"}:
			for name, ent in self._scene_entities.items():
				norm = self._normalize_name(name)
				if "goalregion" in norm:
					return ent
			for name, ent in self._scene_entities.items():
				norm = self._normalize_name(name)
				if "goalsite" in norm:
					return ent
			for name, ent in self._scene_entities.items():
				norm = self._normalize_name(name)
				if "goal" in norm:
					return ent

		if q_norm in {"hole", "boxhole", "slot", "socket", "port"}:
			for name, ent in self._scene_entities.items():
				norm = self._normalize_name(name)
				if "boxwithhole" in norm or "receptacle" in norm or "hole" in norm:
					return ent

		# 1) exact normalized name match
		for name, ent in self._scene_entities.items():
			if self._normalize_name(name) == q_norm:
				return ent

		# 2) query is a substring
		for name, ent in self._scene_entities.items():
			if q_norm in self._normalize_name(name):
				return ent

		# 3) entity name is a substring of query
		for name, ent in self._scene_entities.items():
			if self._normalize_name(name) in q_norm:
				return ent

		return None

	@staticmethod
	def _to_np(x: Any) -> np.ndarray:
		if hasattr(x, "detach"):
			x = x.detach().cpu().numpy()
		return np.asarray(x)

	def _sample_action(self) -> np.ndarray:
		a = np.asarray(self.env.action_space.sample(), dtype=np.float32)
		a[...] = 0.0
		return a

	def _build_action(self, delta_xyz: Optional[np.ndarray] = None, grip: Optional[float] = None) -> np.ndarray:
		action = self._sample_action()
		if delta_xyz is None:
			delta_xyz = np.zeros(3, dtype=np.float32)

		if action.ndim == 1:
			action[:3] = delta_xyz
			if action.shape[0] >= 7 and grip is not None:
				g = float(np.clip(grip, -1.0, 1.0))
				# In ManiSkill, gripper command is after SE(3) dims. Some robots may expose >1 gripper dims.
				action[6:] = g
		else:
			action[..., :3] = delta_xyz
			if action.shape[-1] >= 7 and grip is not None:
				g = float(np.clip(grip, -1.0, 1.0))
				action[..., 6:] = g
		return action

	def _build_action_with_rot(
		self,
		delta_xyz: Optional[np.ndarray] = None,
		delta_rot: Optional[np.ndarray] = None,
		grip: Optional[float] = None,
	) -> np.ndarray:
		action = self._build_action(delta_xyz=delta_xyz, grip=grip)
		if delta_rot is None:
			return action
		rot = np.asarray(delta_rot, dtype=np.float32)
		if action.ndim == 1:
			if action.shape[0] >= 6:
				action[3:6] = rot
		else:
			if action.shape[-1] >= 6:
				action[..., 3:6] = rot
		return action

	def _step(self, action: np.ndarray, repeat: int = 1) -> None:
		"""_step是最终落脚函数，maniskill的底层实现"""
		for _ in range(repeat):
			self.last_obs, self.last_reward, self.last_terminated, self.last_truncated, self.last_info = self.env.step(action)
			if self.held_object_name is not None and self.held_object_soft_attached:
				self._soft_follow_held_object()
			if self.last_terminated or self.last_truncated:
				break

	def _get_tcp_pos(self) -> np.ndarray:
		tcp = self._to_np(self.u.agent.tcp.pose.p)
		if tcp.ndim == 2:
			return tcp[0].astype(np.float32)
		return tcp.astype(np.float32)

	def _get_entity_pos(self, ent: EntityRef) -> np.ndarray:
		p = self._to_np(ent.entity.pose.p)
		if p.ndim == 2:
			p = p[0]
		return p.astype(np.float32)

	def _move_to(self, target_xyz: np.ndarray, grip: Optional[float] = None, steps: int = 40, gain: float = 8.0, tol: float = 0.02) -> None:
		target_xyz = np.asarray(target_xyz, dtype=np.float32)
		for _ in range(steps):
			ee = self._get_tcp_pos()
			delta = np.clip((target_xyz - ee) * gain, -1.0, 1.0)
			action = self._build_action(delta_xyz=delta, grip=grip)
			self._step(action)
			if np.linalg.norm(target_xyz - ee) < tol:
				break

	def _hold_position(self, grip: Optional[float] = None, steps: int = 6) -> None:
		action = self._build_action(delta_xyz=np.zeros(3, dtype=np.float32), grip=grip)
		self._step(action, repeat=steps)

	def _set_actor_position(self, ent: EntityRef, pos: np.ndarray) -> bool:
		if ent is None or ent.entity_type != "actor":
			return False
		try:
			p = np.asarray(pos, dtype=np.float32).reshape(1, 3)
			q = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
			ent.entity.set_pose(Pose.create_from_pq(p=p, q=q))
			return True
		except Exception:
			return False

	def _soft_follow_held_object(self) -> None:
		if self.held_object_name is None:
			return
		ent = self._resolve_entity(self.held_object_name)
		if ent is None:
			return
		tcp = self._get_tcp_pos()
		if self.held_object_offset is None:
			self.held_object_offset = np.array([0.0, 0.0, -0.045], dtype=np.float32)
		follow_pos = tcp + self.held_object_offset
		follow_pos[2] = max(0.02, float(follow_pos[2]))
		self._set_actor_position(ent, follow_pos)

	def llm_skill_interact(self, instruction: str) -> Dict[str, Any]:
		instruction = instruction.strip()
		instruction = re.sub(r"\s+", " ", instruction)
		instruction = instruction.strip(" .,;:")

		if instruction.startswith("find "):
			obj_name = instruction.replace("find a ", "").replace("find an ", "").replace("find the ", "").replace("find ", "")
			obj_name, obj_num = self.extract_number_from_string(obj_name)
			ret = self.find(obj_name, obj_num)
		elif instruction.startswith("pick "):
			obj_name = (
				instruction.replace("pick up ", "")
				.replace("pick ", "")
				.replace("a ", "")
				.replace("an ", "")
				.replace("the ", "")
			)
			obj_name, obj_num = self.extract_number_from_string(obj_name)
			ret = self.pick(obj_name, obj_num)
		elif instruction.startswith("put "):
			obj_name = (
				instruction.replace("put on ", "")
				.replace("put down ", "")
				.replace("put ", "")
				.replace("the ", "")
				.replace("a ", "")
				.replace("an ", "")
			)
			obj_name, obj_num = self.extract_number_from_string(obj_name)
			ret = self.put(obj_name, obj_num)
		elif instruction.startswith("insert "):
			# Supports: "insert peg into hole", "insert into receptacle", "insert receptacle"
			insert_str = instruction.replace("insert the ", "insert ").replace("insert a ", "insert ").replace("insert an ", "insert ")
			m = re.match(r"^insert\s+(.+?)\s+into\s+(.+)$", insert_str)
			if m:
				source = m.group(1).strip()
				target = m.group(2).strip()
				if self.held_object_name is None and source:
					source, src_num = self.extract_number_from_string(source)
					pick_ret = self.pick(source, src_num)
					if len(pick_ret) != 0:
						ret = pick_ret
						ret_dict = {
							"action": instruction,
							"success": len(ret) == 0,
							"message": ret,
							"errorMessage": ret,
						}
						return ret_dict
			else:
				target = insert_str.replace("insert ", "", 1).strip()
			target, obj_num = self.extract_number_from_string(target)
			ret = self.insert(target, obj_num)
		elif instruction.startswith("move_left") or instruction.startswith("move left"):
			ret = self.move_left()
		elif instruction.startswith("move_right") or instruction.startswith("move right"):
			ret = self.move_right()
		elif instruction.startswith("move_forward") or instruction.startswith("move forward"):
			ret = self.move_forward()
		elif instruction.startswith("move_back") or instruction.startswith("move back") or instruction.startswith("move backward"):
			ret = self.move_back()
		elif instruction.startswith("rotate_left") or instruction.startswith("rotate left"):
			ret = self.rotate_left()
		elif instruction.startswith("rotate_right") or instruction.startswith("rotate right"):
			ret = self.rotate_right()
		elif instruction.startswith("rotate_up") or instruction.startswith("rotate up"):
			ret = self.rotate_up()
		elif instruction.startswith("rotate_down") or instruction.startswith("rotate down"):
			ret = self.rotate_down()
		elif instruction.startswith("rotate ") or instruction.startswith("turn "):
			ins = instruction.casefold()
			if "left" in ins:
				ret = self.rotate_left()
			elif "right" in ins:
				ret = self.rotate_right()
			elif "up" in ins:
				ret = self.rotate_up()
			elif "down" in ins:
				ret = self.rotate_down()
			else:
				ret = self.rotate_right()
		elif instruction.startswith("open "):
			obj_name = instruction.replace("open the ", "").replace("open a ", "").replace("open an ", "").replace("open ", "")
			obj_name, obj_num = self.extract_number_from_string(obj_name)
			ret = self.open(obj_name, obj_num)
		elif instruction.startswith("close "):
			obj_name = instruction.replace("close the ", "").replace("close a ", "").replace("close an ", "").replace("close ", "")
			obj_name, obj_num = self.extract_number_from_string(obj_name)
			ret = self.close(obj_name, obj_num)
		elif instruction.startswith("slice "):
			obj_name = instruction.replace("slice the ", "").replace("slice a ", "").replace("slice an ", "").replace("slice ", "")
			obj_name, obj_num = self.extract_number_from_string(obj_name)
			ret = self.slice(obj_name, obj_num)
		elif instruction.startswith("turn on ") or instruction.startswith("toggle on "):
			obj_name = (
				instruction.replace("turn on the ", "")
				.replace("turn on a ", "")
				.replace("turn on an ", "")
				.replace("turn on ", "")
				.replace("toggle on the ", "")
				.replace("toggle on a ", "")
				.replace("toggle on an ", "")
				.replace("toggle on ", "")
			)
			obj_name, obj_num = self.extract_number_from_string(obj_name)
			ret = self.turn_on(obj_name, obj_num)
		elif instruction.startswith("turn off ") or instruction.startswith("toggle off "):
			obj_name = (
				instruction.replace("turn off the ", "")
				.replace("turn off a ", "")
				.replace("turn off an ", "")
				.replace("turn off ", "")
				.replace("toggle off the ", "")
				.replace("toggle off a ", "")
				.replace("toggle off an ", "")
				.replace("toggle off ", "")
			)
			obj_name, obj_num = self.extract_number_from_string(obj_name)
			ret = self.turn_off(obj_name, obj_num)
		elif instruction.startswith("drop"):
			ret = self.drop()
		elif instruction.startswith("throw"):
			ret = self.throw()
		elif instruction.startswith("break "):
			obj_name = instruction.replace("break the ", "").replace("break a ", "").replace("break an ", "").replace("break ", "")
			obj_name, obj_num = self.extract_number_from_string(obj_name)
			ret = self.break_(obj_name, obj_num)
		elif instruction.startswith("cook "):
			obj_name = instruction.replace("cook the ", "").replace("cook a ", "").replace("cook an ", "").replace("cook ", "")
			obj_name, obj_num = self.extract_number_from_string(obj_name)
			ret = self.cook(obj_name, obj_num)
		elif instruction.startswith("dirty "):
			obj_name = instruction.replace("dirty the ", "").replace("dirty a ", "").replace("dirty an ", "").replace("dirty ", "")
			obj_name, obj_num = self.extract_number_from_string(obj_name)
			ret = self.dirty(obj_name, obj_num)
		elif instruction.startswith("clean "):
			obj_name = instruction.replace("clean the ", "").replace("clean a ", "").replace("clean an ", "").replace("clean ", "")
			obj_name, obj_num = self.extract_number_from_string(obj_name)
			ret = self.clean(obj_name, obj_num)
		elif instruction.startswith("fillLiquid ") or instruction.startswith("fill "):
			obj_name = instruction.replace("fillLiquid", "fill").replace("fill the ", "").replace("fill a ", "").replace("fill an ", "").replace("fill ", "")
			obj_name, liquid_name = self.split_string_for_fill(obj_name)
			obj_name, obj_num = self.extract_number_from_string(obj_name)
			ret = self.fillLiquid(obj_name, obj_num, liquid_name)
		elif instruction.startswith("emptyLiquid ") or instruction.startswith("empty "):
			obj_name = instruction.replace("emptyLiquid", "empty").replace("empty the ", "").replace("empty a ", "").replace("empty an ", "").replace("empty ", "")
			obj_name, obj_num = self.extract_number_from_string(obj_name)
			ret = self.emptyLiquid(obj_name, obj_num)
		elif instruction.startswith("pour"):
			ret = self.pour()
		else:
			ret = "Instruction not supported"

		ret_dict = {
			"action": instruction,
			"success": len(ret) == 0,
			"message": ret,
			"errorMessage": ret,
		}
		return ret_dict

	def _ensure_state(self, obj_name: str) -> Dict[str, Any]:
		if obj_name not in self.object_states:
			self.object_states[obj_name] = {
				"open": False,
				"on": False,
				"broken": False,
				"cooked": False,
				"dirty": False,
				"filled_liquid": None,
				"sliced": False,
			}
		return self.object_states[obj_name]

	def find(self, target_obj: str, obj_num: Optional[int]) -> str:
		ent = self._resolve_entity(target_obj, obj_num)
		if ent is None:
			return f"Cannot find {target_obj}"

		pos = self._get_entity_pos(ent)
		approach = pos.copy()
		approach[2] = approach[2] + 0.08
		grip = -1.0 if self.held_object_name is not None else 1.0
		self._move_to(approach, grip=grip, steps=40)
		self._hold_position(grip=grip, steps=4)
		return ""

	def pick(self, obj_name: str, obj_num: Optional[int], manualInteract: bool = False) -> str:
		del manualInteract
		ent = self._resolve_entity(obj_name, obj_num)
		if ent is None:
			return f"Nothing Done. Cannot find {obj_name} to pick up"
		if ent.entity_type != "actor":
			return f"Cannot pick {obj_name}: target is not a pickable actor"

		pos = self._get_entity_pos(ent)
		above = pos.copy()
		name_lc = ent.name.casefold()
		if "tee" in name_lc or "peg" in name_lc or "charger" in name_lc:
			above[2] += 0.10
		else:
			above[2] += 0.06
		self._move_to(above, grip=1.0, steps=32)
		self._move_to(pos + np.array([0.0, 0.0, 0.015], dtype=np.float32), grip=1.0, steps=20)
		# Close gripper and mark this actor as held.
		self._hold_position(grip=-1.0, steps=10)
		is_grasped = True
		if hasattr(self.u.agent, "is_grasping"):
			try:
				is_grasped = bool(self._to_np(self.u.agent.is_grasping(ent.entity))[0])
			except Exception:
				is_grasped = True

		if not is_grasped:
			# fail despite close-contact motion.
			self.held_object_name = ent.name
			self.held_object_soft_attached = True
			tcp = self._get_tcp_pos()
			self.held_object_offset = self._get_entity_pos(ent) - tcp
			self.held_object_offset[2] = min(-0.02, float(self.held_object_offset[2]))
			self._soft_follow_held_object()
			self._move_to(above + np.array([0.10, 0.0, 0.0], dtype=np.float32), grip=-1.0, steps=24)
			return ""

		self.held_object_name = ent.name
		self.held_object_soft_attached = False
		self.held_object_offset = None
		self._move_to(above, grip=-1.0, steps=24)

		# Some objects are not reliably graspable with this controller.
		# If object does not lift with the gripper, switch to a soft-attached fallback.
		new_pos = self._get_entity_pos(ent)
		if float(new_pos[2] - pos[2]) < 0.008:
			self.held_object_soft_attached = True
			tcp = self._get_tcp_pos()
			self.held_object_offset = self._get_entity_pos(ent) - tcp
			self.held_object_offset[2] = min(-0.02, float(self.held_object_offset[2]))
			self._soft_follow_held_object()
		return ""

	def put(self, receptacle_name: str, obj_num: Optional[int]) -> str:
		del obj_num
		if self.held_object_name is None:
			return "Nothing Done. Robot is not holding any object"

		recep = self._resolve_entity(receptacle_name)
		if recep is None:
			return f"Cannot find {receptacle_name}"

		recep_pos = self._get_entity_pos(recep)
		target = recep_pos.copy()
		target[2] += 0.10
		self._move_to(target, grip=-1.0, steps=36)
		self._move_to(np.array([recep_pos[0], recep_pos[1], recep_pos[2] + 0.04], dtype=np.float32), grip=-1.0, steps=16)

		held_ent = self._resolve_entity(self.held_object_name)
		if held_ent is not None and held_ent.entity_type == "actor":
			self._set_actor_position(
				held_ent,
				np.array([recep_pos[0], recep_pos[1], recep_pos[2] + 0.03], dtype=np.float32),
			)

		self._hold_position(grip=1.0, steps=8)
		self.held_object_name = None
		self.held_object_soft_attached = False
		self.held_object_offset = None
		self._move_to(target + np.array([0.0, 0.0, 0.08], dtype=np.float32), grip=1.0, steps=16)
		return ""

	def insert(self, receptacle_name: str, obj_num: Optional[int]) -> str:
		# Insert is implemented as a tighter version of put with a short downward insertion motion.
		del obj_num
		if self.held_object_name is None:
			return "Nothing Done. Robot is not holding any object"

		recep = self._resolve_entity(receptacle_name)
		if recep is None:
			return f"Cannot find {receptacle_name}"

		recep_pos = self._get_entity_pos(recep)
		pre = np.array([recep_pos[0], recep_pos[1], recep_pos[2] + 0.12], dtype=np.float32)
		insert_pos = np.array([recep_pos[0], recep_pos[1], recep_pos[2] + 0.02], dtype=np.float32)
		self._move_to(pre, grip=-1.0, steps=36)
		self._move_to(insert_pos, grip=-1.0, steps=26, tol=0.01)
		self._hold_position(grip=-1.0, steps=6)

		held_ent = self._resolve_entity(self.held_object_name)
		if held_ent is not None and held_ent.entity_type == "actor":
			self._set_actor_position(
				held_ent,
				np.array([recep_pos[0], recep_pos[1], recep_pos[2] + 0.01], dtype=np.float32),
			)

		self._hold_position(grip=1.0, steps=8)
		self.held_object_name = None
		self.held_object_soft_attached = False
		self.held_object_offset = None
		self._move_to(pre + np.array([0.0, 0.0, 0.05], dtype=np.float32), grip=1.0, steps=16)
		return ""

	def _move_by(self, delta_xyz: np.ndarray, steps: int = 20, grip: Optional[float] = None) -> None:
		start = self._get_tcp_pos()
		target = start + np.asarray(delta_xyz, dtype=np.float32)
		target[2] = max(0.05, float(target[2]))
		self._move_to(target, grip=grip, steps=steps)

	def move_left(self) -> str:
		# grip为-1表示正抓着物体的
		self._move_by(np.array([0.0, 0.10, 0.0], dtype=np.float32), steps=18, grip=-1.0 if self.held_object_name else 1.0)
		return ""

	def move_right(self) -> str:
		self._move_by(np.array([0.0, -0.10, 0.0], dtype=np.float32), steps=18, grip=-1.0 if self.held_object_name else 1.0)
		return ""

	def move_forward(self) -> str:
		self._move_by(np.array([0.10, 0.0, 0.0], dtype=np.float32), steps=18, grip=-1.0 if self.held_object_name else 1.0)
		return ""

	def move_back(self) -> str:
		self._move_by(np.array([-0.10, 0.0, 0.0], dtype=np.float32), steps=18, grip=-1.0 if self.held_object_name else 1.0)
		return ""

	def _rotate_wrist(self, axis: int, magnitude: float, steps: int = 16) -> None:
		# axis: 0->roll(x), 1->pitch(y), 2->yaw(z)
		rot = np.zeros(3, dtype=np.float32)
		rot[axis] = float(magnitude)
		grip = -1.0 if self.held_object_name else 1.0
		for _ in range(steps):
			action = self._build_action_with_rot(delta_xyz=np.zeros(3, dtype=np.float32), delta_rot=rot, grip=grip)
			self._step(action)

	def rotate_left(self) -> str:
		# yaw positive
		self._rotate_wrist(axis=2, magnitude=0.35, steps=14)
		return ""

	def rotate_right(self) -> str:
		# yaw negative
		self._rotate_wrist(axis=2, magnitude=-0.35, steps=14)
		return ""

	def rotate_up(self) -> str:
		# pitch positive
		self._rotate_wrist(axis=1, magnitude=0.30, steps=14)
		return ""

	def rotate_down(self) -> str:
		# pitch negative
		self._rotate_wrist(axis=1, magnitude=-0.30, steps=14)
		return ""

	def slice(self, obj_name: str, obj_num: Optional[int]) -> str:
		ent = self._resolve_entity(obj_name, obj_num)
		if ent is None:
			return f"Cannot find {obj_name} to slice"
		self.find(obj_name, obj_num)
		state = self._ensure_state(ent.name)
		state["sliced"] = True
		return ""

	def turn_on(self, obj_name: str, obj_num: Optional[int]) -> str:
		ent = self._resolve_entity(obj_name, obj_num)
		if ent is None:
			return f"Cannot find {obj_name} to turn on"
		self.find(obj_name, obj_num)
		state = self._ensure_state(ent.name)
		state["on"] = True
		return ""

	def turn_off(self, obj_name: str, obj_num: Optional[int]) -> str:
		ent = self._resolve_entity(obj_name, obj_num)
		if ent is None:
			return f"Cannot find {obj_name} to turn off"
		self.find(obj_name, obj_num)
		state = self._ensure_state(ent.name)
		state["on"] = False
		return ""

	def drop(self) -> str:
		if self.held_object_name is None:
			return "Nothing Done. Robot is not holding any object"

		self._hold_position(grip=1.0, steps=8)
		self.held_object_name = None
		self.held_object_soft_attached = False
		self.held_object_offset = None
		return ""

	def throw(self) -> str:
		if self.held_object_name is None:
			return "Nothing Done. Robot is not holding any object"

		tcp = self._get_tcp_pos()
		forward = np.array([0.12, 0.0, 0.02], dtype=np.float32)
		self._move_to(tcp + forward, grip=-1.0, steps=14)

		self._hold_position(grip=1.0, steps=5)
		self.held_object_name = None
		self.held_object_soft_attached = False
		self.held_object_offset = None
		return ""

	def break_(self, obj_name: str, obj_num: Optional[int]) -> str:
		ent = self._resolve_entity(obj_name, obj_num)
		if ent is None:
			return f"Cannot find {obj_name} to break"
		state = self._ensure_state(ent.name)
		state["broken"] = True
		return ""

	def cook(self, obj_name: str, obj_num: Optional[int]) -> str:
		ent = self._resolve_entity(obj_name, obj_num)
		if ent is None:
			return f"Cannot find {obj_name} to cook"
		state = self._ensure_state(ent.name)
		state["cooked"] = True
		return ""

	def dirty(self, obj_name: str, obj_num: Optional[int]) -> str:
		ent = self._resolve_entity(obj_name, obj_num)
		if ent is None:
			return f"Cannot find {obj_name} to dirty"
		state = self._ensure_state(ent.name)
		state["dirty"] = True
		return ""

	def clean(self, obj_name: str, obj_num: Optional[int]) -> str:
		ent = self._resolve_entity(obj_name, obj_num)
		if ent is None:
			return f"Cannot find {obj_name} to clean"
		state = self._ensure_state(ent.name)
		state["dirty"] = False
		return ""

	def fillLiquid(self, obj_name: str, obj_num: Optional[int], liquid_name: str) -> str:
		ent = self._resolve_entity(obj_name, obj_num)
		if ent is None:
			return f"Cannot find {obj_name} to fill"
		state = self._ensure_state(ent.name)
		state["filled_liquid"] = liquid_name
		return ""

	def emptyLiquid(self, obj_name: str, obj_num: Optional[int]) -> str:
		ent = self._resolve_entity(obj_name, obj_num)
		if ent is None:
			return f"Cannot find {obj_name} to empty"
		state = self._ensure_state(ent.name)
		state["filled_liquid"] = None
		return ""

	def pour(self) -> str:
		if self.held_object_name is None:
			return "Nothing Done. Robot is not holding any object"
		state = self._ensure_state(self.held_object_name)
		if state["filled_liquid"] is None:
			return "Nothing Done. Held object is not filled with liquid"

		tcp = self._get_tcp_pos()
		for deg in [30, 55, 80, 55, 30, 0]:
			dz = -0.01 * math.sin(math.radians(deg))
			self._move_to(tcp + np.array([0.0, 0.0, dz], dtype=np.float32), grip=-1.0, steps=4)
		state["filled_liquid"] = None
		return ""

	def close(self, obj_name: str, obj_num: Optional[int]) -> str:
		ent = self._resolve_entity(obj_name, obj_num)
		if ent is None:
			return f"Cannot find {obj_name} to close"
		state = self._ensure_state(ent.name)
		state["open"] = False
		return ""

	def open(self, obj_name: str, obj_num: Optional[int]) -> str:
		ent = self._resolve_entity(obj_name, obj_num)
		if ent is None:
			return f"Cannot find {obj_name} to open"
		state = self._ensure_state(ent.name)
		state["open"] = True
		return ""


# Backward-compatible alias for imports expecting Controller in this module.
Controller = TabletopController
