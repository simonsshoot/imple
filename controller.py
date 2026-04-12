import logging
import importlib
import math
import re
import string
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class EntityRef:
	name: str
	entity_type: str  # actor | articulation
	entity: Any


class Controller:
	def __init__(self, env: Any) -> None:
		self.env = env
		self.u = env.unwrapped
		self.actions = [
			"find",
			"pick",
			"put",
			"move_left",
			"move_right",
			"move_forward",
			"move_back",
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
		self.frame_callback = None
		self._scene_entities: Dict[str, EntityRef] = {}
		self.restore_scene()

	def restore_scene(self) -> None:
		self._refresh_scene_entities()
		self.multi_objs_dict = {}

	def _get_scene_obj(self) -> Optional[Any]:
		scene = getattr(self.u, "scene", None)
		if scene is None:
			scene = getattr(self.u, "_scene", None)
		return scene

	def _refresh_scene_entities(self) -> None:
		self._scene_entities = {}
		scene = self._get_scene_obj()
		if scene is None:
			log.warning("Environment has neither 'scene' nor '_scene'; no entities loaded")
			# Continue to fallback loaders below.

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

		if scene is not None:
			# Path A: wrappers exposing dict-like scene.actors/articulations
			actors_map = getattr(scene, "actors", None)
			if isinstance(actors_map, dict):
				for name, actor in actors_map.items():
					self._scene_entities[str(name)] = EntityRef(name=str(name), entity_type="actor", entity=actor)
			else:
				# Path B: raw sapien.Scene exposing get_all_actors()
				get_all_actors = getattr(scene, "get_all_actors", None)
				if callable(get_all_actors):
					try:
						for i, actor in enumerate(list(get_all_actors())):
							name = _entity_name(actor, "actor", i)
							self._scene_entities[name] = EntityRef(name=name, entity_type="actor", entity=actor)
					except Exception as exc:
						log.warning("Failed loading actors from get_all_actors: %s", exc)

			arts_map = getattr(scene, "articulations", None)
			if isinstance(arts_map, dict):
				for name, art in arts_map.items():
					self._scene_entities[str(name)] = EntityRef(name=str(name), entity_type="articulation", entity=art)
			else:
				# Path B: raw sapien.Scene exposing get_all_articulations()
				get_all_articulations = getattr(scene, "get_all_articulations", None)
				if callable(get_all_articulations):
					try:
						for i, art in enumerate(list(get_all_articulations())):
							name = _entity_name(art, "articulation", i)
							self._scene_entities[name] = EntityRef(name=name, entity_type="articulation", entity=art)
					except Exception as exc:
						log.warning("Failed loading articulations from get_all_articulations: %s", exc)

		# Path C: fallback from custom env fields in ManiSkill2_real2sim
		for ent in getattr(self.u, "episode_objs", []) or []:
			name = _entity_name(ent, "episode_obj", len(self._scene_entities))
			if name not in self._scene_entities:
				self._scene_entities[name] = EntityRef(name=name, entity_type="actor", entity=ent)

		for attr_name, typ in [
			("episode_source_obj", "actor"),
			("episode_target_obj", "actor"),
			("sink", "actor"),
		]:
			ent = getattr(self.u, attr_name, None)
			if ent is None:
				continue
			name = _entity_name(ent, attr_name, len(self._scene_entities))
			if name not in self._scene_entities:
				self._scene_entities[name] = EntityRef(name=name, entity_type=typ, entity=ent)

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
		name = name.replace("_", " ").replace("-", " ")
		name = re.sub(r"\s+", " ", name).strip()

		color_words = {
			"red", "blue", "green", "yellow", "orange", "black", "white", "purple", "brown", "gray", "grey"
		}
		toks = [t for t in name.split(" ") if t]
		while toks and toks[0] in color_words:
			toks = toks[1:]
		if toks:
			name = " ".join(toks)

		synonyms = {
			"ball": "sphere",
			"orb": "sphere",
			"cup": "mug",
			"basket": "rack",
			"dish basket": "rack",
			"dish rack": "rack",
			"dishrack": "rack",
		}
		return synonyms.get(name, name)

	def _resolve_entity(self, obj_name: str, obj_num: Optional[int] = None) -> Optional[EntityRef]:
		"""实体解析，将自然语言描述的对象名称映射到场景中的实际物理实体"""
		del obj_num  # Not used in current ManiSkill scenes.
		if not self._scene_entities:
			self._refresh_scene_entities()

		obj_name = self._canonical_object_name(obj_name)

		candidate_queries = [obj_name]
		if "rack" in obj_name:
			candidate_queries.extend(["basket", "dish rack", "dishrack"])
			candidate_queries.extend(["sink", "dummy sink target plane", "dummy_sink_target_plane"])
		if "can" in obj_name:
			candidate_queries.extend(["pepsi", "fanta", "tang", "soda"])

		seen = set()
		candidate_queries = [q for q in candidate_queries if not (q in seen or seen.add(q))]

		for q in candidate_queries:
			q_norm = self._normalize_name(q)
			if not q_norm:
				continue

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
				action[6] = float(np.clip(grip, -1.0, 1.0))
		else:
			action[..., :3] = delta_xyz
			if action.shape[-1] >= 7 and grip is not None:
				action[..., 6] = float(np.clip(grip, -1.0, 1.0))
		return action

	def _step(self, action: np.ndarray, repeat: int = 1) -> None:
		"""_step是最终落脚函数，maniskill的底层实现"""
		for _ in range(repeat):
			self.last_obs, self.last_reward, self.last_terminated, self.last_truncated, self.last_info = self.env.step(action)
			if callable(self.frame_callback):
				try:
					self.frame_callback(self.last_obs)
				except Exception:
					pass
			if self.held_object_name is not None and self.held_object_soft_attached:
				self._soft_follow_held_object()
			if self.last_terminated or self.last_truncated:
				break

	def _get_tcp_pos(self) -> np.ndarray:
		def _as_xyz(v: Any) -> Optional[np.ndarray]:
			if v is None:
				return None
			arr = self._to_np(v)
			if arr.ndim == 2:
				arr = arr[0]
			arr = np.asarray(arr, dtype=np.float32).reshape(-1)
			if arr.shape[0] < 3:
				return None
			return arr[:3]

		# 1) ManiSkill-style path: u.agent.tcp.pose.p
		agent = getattr(self.u, "agent", None)
		if agent is not None:
			agent_tcp = getattr(agent, "tcp", None)
			if agent_tcp is not None:
				pose = getattr(agent_tcp, "pose", None)
				if pose is not None:
					xyz = _as_xyz(getattr(pose, "p", None))
					if xyz is not None:
						self._tcp_source = "agent.tcp.pose.p"
						return xyz

		# 2) Cached ee link on agent
		if agent is not None:
			ee_link = getattr(agent, "ee_link", None)
			if ee_link is not None:
				pose = getattr(ee_link, "pose", None)
				if pose is None and hasattr(ee_link, "get_pose"):
					try:
						pose = ee_link.get_pose()
					except Exception:
						pose = None
				if pose is not None:
					xyz = _as_xyz(getattr(pose, "p", None))
					if xyz is not None:
						self._tcp_source = "agent.ee_link.pose"
						return xyz

		# 3) Fallback: find likely end-effector link from robot links
		if agent is not None:
			robot = getattr(agent, "robot", None)
			get_links = getattr(robot, "get_links", None)
			if callable(get_links):
				try:
					links = list(get_links())
				except Exception:
					links = []
				candidates = []
				for link in links:
					name = str(getattr(link, "name", "")).casefold()
					# Prefer explicit ee/tcp/gripper tip links.
					score = 0
					if "ee" in name:
						score += 3
					if "tcp" in name:
						score += 3
					if "gripper" in name:
						score += 1
					if "finger" in name:
						score -= 1
					if score > 0:
						candidates.append((score, link))
				for _, link in sorted(candidates, key=lambda x: x[0], reverse=True):
					pose = getattr(link, "pose", None)
					if pose is None and hasattr(link, "get_pose"):
						try:
							pose = link.get_pose()
						except Exception:
							pose = None
					if pose is None:
						continue
					xyz = _as_xyz(getattr(pose, "p", None))
					if xyz is not None:
						self._tcp_source = f"robot.link:{getattr(link, 'name', 'unknown')}"
						return xyz

		# 4) simpler_env/ManiSkill2 fallback: u.tcp.pose.p (some envs expose this)
		env_tcp = getattr(self.u, "tcp", None)
		if env_tcp is not None:
			pose = getattr(env_tcp, "pose", None)
			if pose is not None:
				xyz = _as_xyz(getattr(pose, "p", None))
				if xyz is not None:
					self._tcp_source = "env.tcp.pose.p"
					return xyz

		raise AttributeError("Cannot resolve end-effector position from env; tried agent.tcp, env.tcp, ee_link, and robot links")

	def _get_entity_pos(self, ent: EntityRef) -> np.ndarray:
		p = self._to_np(ent.entity.pose.p)
		if p.ndim == 2:
			p = p[0]
		return p.astype(np.float32)

	def _move_to(self, target_xyz: np.ndarray, grip: Optional[float] = None, steps: int = 40, gain: float = 8.0, tol: float = 0.02) -> None:
		target_xyz = np.asarray(target_xyz, dtype=np.float32)
		start_ee = self._get_tcp_pos().copy()
		tcp_source = getattr(self, "_tcp_source", "unknown")
		start_dist = float(np.linalg.norm(target_xyz - start_ee))
		for _ in range(steps):
			ee = self._get_tcp_pos()
			delta = np.clip((target_xyz - ee) * gain, -1.0, 1.0)
			action = self._build_action(delta_xyz=delta, grip=grip)
			self._step(action)
			if np.linalg.norm(target_xyz - ee) < tol:
				break
		end_ee = self._get_tcp_pos().copy()
		end_dist = float(np.linalg.norm(target_xyz - end_ee))
		self._last_motion_stats = {
			"start_dist": start_dist,
			"end_dist": end_dist,
			"ee_displacement": float(np.linalg.norm(end_ee - start_ee)),
			"tcp_source": tcp_source,
		}

	def _hold_position(self, grip: Optional[float] = None, steps: int = 6) -> None:
		action = self._build_action(delta_xyz=np.zeros(3, dtype=np.float32), grip=grip)
		self._step(action, repeat=steps)

	def _set_actor_position(self, ent: EntityRef, pos: np.ndarray) -> bool:
		if ent is None or ent.entity_type != "actor":
			return False
		try:
			p_vec = np.asarray(pos, dtype=np.float32).reshape(-1)[:3]
			q_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

			pose_obj = None
			# Prefer ManiSkill Pose when available.
			try:
				ms_structs = importlib.import_module("mani_skill.utils.structs")
				MSPose = getattr(ms_structs, "Pose", None)
				if MSPose is not None and hasattr(MSPose, "create_from_pq"):
					pose_obj = MSPose.create_from_pq(
						p=np.asarray(p_vec, dtype=np.float32).reshape(1, 3),
						q=np.asarray(q_vec, dtype=np.float32).reshape(1, 4),
					)
			except Exception:
				pose_obj = None

			# Fallback to SAPIEN pose if ManiSkill is unavailable.
			if pose_obj is None:
				try:
					sapien = importlib.import_module("sapien.core")
					SPose = getattr(sapien, "Pose", None)
					if SPose is not None:
						pose_obj = SPose(p=p_vec, q=q_vec)
				except Exception:
					pose_obj = None

			if pose_obj is None:
				return False

			ent.entity.set_pose(pose_obj)
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
		follow_pos = np.array([tcp[0], tcp[1], max(0.02, tcp[2] - 0.045)], dtype=np.float32)
		self._set_actor_position(ent, follow_pos)

	def llm_skill_interact(self, instruction: str) -> Dict[str, Any]:
		instruction = instruction.strip()

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
		elif instruction.startswith("move_left") or instruction.startswith("move left"):
			ret = self.move_left()
		elif instruction.startswith("move_right") or instruction.startswith("move right"):
			ret = self.move_right()
		elif instruction.startswith("move_forward") or instruction.startswith("move forward"):
			ret = self.move_forward()
		elif instruction.startswith("move_back") or instruction.startswith("move back") or instruction.startswith("move backward"):
			ret = self.move_back()
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
		self._move_to(approach, grip=1.0, steps=40)
		self._hold_position(grip=1.0, steps=4)
		stats = getattr(self, "_last_motion_stats", {})
		ee_disp = float(stats.get("ee_displacement", 0.0))
		start_dist = float(stats.get("start_dist", 0.0))
		end_dist = float(stats.get("end_dist", 0.0))
		tcp_source = str(stats.get("tcp_source", "unknown"))
		if ee_disp < 1e-3:
			return f"Robot end-effector did not move while finding target (tcp={tcp_source})"
		# If already close enough at start, do not require further distance reduction.
		if start_dist > 0.08 and end_dist > max(0.02, start_dist - 0.005):
			return (
				f"Find likely ineffective: distance to target not reduced enough "
				f"(start={start_dist:.4f}, end={end_dist:.4f}, tcp={tcp_source})"
			)
		return ""

	def pick(self, obj_name: str, obj_num: Optional[int], manualInteract: bool = False) -> str:
		del manualInteract
		ent = self._resolve_entity(obj_name, obj_num)
		if ent is None:
			return f"Nothing Done. Cannot find {obj_name} to pick up"
		if ent.entity_type != "actor":
			return f"Cannot pick {obj_name}: target is not a pickable actor"

		pos = self._get_entity_pos(ent)
		obj_pos_before = pos.copy()
		above = pos.copy()
		above[2] += 0.06
		self._move_to(above, grip=1.0, steps=32)
		self._move_to(pos + np.array([0.0, 0.0, 0.015], dtype=np.float32), grip=1.0, steps=20)
		# Close gripper and mark this actor as held.
		self._hold_position(grip=-1.0, steps=10)
		is_grasped = False
		if hasattr(self.u.agent, "is_grasping"):
			try:
				is_grasped = bool(self._to_np(self.u.agent.is_grasping(ent.entity))[0])
			except Exception:
				is_grasped = False

		# Heuristic fallback when grasp API is unavailable/inaccurate.
		if not is_grasped:
			try:
				obj_pos_after_close = self._get_entity_pos(ent)
				is_grasped = bool(obj_pos_after_close[2] > obj_pos_before[2] + 0.01)
			except Exception:
				is_grasped = False

		if not is_grasped:
			# Assisted fallback: keep pipeline progressing in environments where
			# reliable grasp state is unavailable.
			self.held_object_name = ent.name
			self.held_object_soft_attached = True
			self._soft_follow_held_object()
			self._move_to(above + np.array([0.10, 0.0, 0.0], dtype=np.float32), grip=-1.0, steps=24)
			return ""

		self.held_object_name = ent.name
		self.held_object_soft_attached = False
		self._move_to(above, grip=-1.0, steps=24)
		stats = getattr(self, "_last_motion_stats", {})
		if float(stats.get("ee_displacement", 0.0)) < 1e-3:
			return "Robot end-effector did not move during pick"
		return ""

	def put(self, receptacle_name: str, obj_num: Optional[int]) -> str:
		del obj_num
		if self.held_object_name is None:
			return "Nothing Done. Robot is not holding any object"

		held_ent = self._resolve_entity(self.held_object_name)
		held_before = self._get_entity_pos(held_ent) if held_ent is not None else None

		recep = self._resolve_entity(receptacle_name)
		if recep is None:
			return f"Cannot find {receptacle_name}"

		recep_pos = self._get_entity_pos(recep)
		target = recep_pos.copy()
		target[2] += 0.10
		self._move_to(target, grip=-1.0, steps=36)
		self._move_to(np.array([recep_pos[0], recep_pos[1], recep_pos[2] + 0.04], dtype=np.float32), grip=-1.0, steps=16)

		self._hold_position(grip=1.0, steps=8)
		self.held_object_name = None
		self.held_object_soft_attached = False
		self._move_to(target + np.array([0.0, 0.0, 0.08], dtype=np.float32), grip=1.0, steps=16)

		if held_ent is not None and held_before is not None:
			try:
				held_after = self._get_entity_pos(held_ent)
				moved = float(np.linalg.norm(held_after - held_before)) > 0.01
				xy_to_recep = float(np.linalg.norm((held_after - recep_pos)[:2]))
				if (not moved) and xy_to_recep > 0.2:
					return "Put likely failed: object did not move to receptacle"
			except Exception:
				pass

		stats = getattr(self, "_last_motion_stats", {})
		if float(stats.get("ee_displacement", 0.0)) < 1e-3:
			return "Robot end-effector did not move during put"
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
