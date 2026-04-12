import ast
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils import ndarray_to_base64, model_selection


class Agents:
  def __init__(self, image: Any, task_description: str, model: str = "gpt-4o") -> None:
    self.image = image
    self.encoded_image = ndarray_to_base64(image)
    self.task_description = task_description
    self.client, self.model = model_selection(model)
    # Low-level action generation is fixed to DeepSeek.
    self.low_level_client, self.low_level_model = model_selection("deepseek-chat")

  def multi_agent_vision_planning(self, objs_from_scene: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, str]:
    def filter_task_objects(objs: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
      if not objs:
        return objs
      task_words = set(self.task_description.lower().replace("-", " ").split())
      always_keep = {"sphere", "ball", "cube", "table", "workspace", "robot", "panda", "arm"}

      filtered: List[Dict[str, Any]] = []
      for obj in objs:
        if not isinstance(obj, dict):
          continue
        name = str(obj.get("name", ""))
        tokens = set(name.lower().replace("_", " ").replace("-", " ").split())
        if (tokens & task_words) or (tokens & always_keep):
          filtered.append(obj)
      return filtered if filtered else objs

    task_objs = filter_task_objects(objs_from_scene)

    def sim_ground_agent() -> str:
      prompt = (
        "You are an assistant that describes the scene and the task-relevant entities. "
        "Use only objects in this metadata when possible: "
        f"{task_objs}. "
        "Then provide concise task-solving instructions."
      )
      resp = self.client.chat.completions.create(
        model=self.model,
        messages=[
          {
            "role": "user",
            "content": [
              {"type": "text", "text": prompt + " Task: " + self.task_description},
              {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self.encoded_image}"}},
            ],
          }
        ],
        max_tokens=500,
        temperature=0,
      )
      text = resp.choices[0].message.content
      print("[sim_ground_agent] response:", text)
      return text

    environment_info = sim_ground_agent()
    planner = self.client.chat.completions.create(
      model=self.model,
      messages=[
        {
          "role": "system",
          "content": (
            "You produce a concise high-level action plan for robot manipulation. "
            "Output only step lines, no extra commentary."
          ),
        },
        {
          "role": "user",
          "content": (
            "Task: " + self.task_description + "\n"
            "Scene context:\n" + environment_info + "\n"
            "Return atomic high-level steps."
          ),
        },
      ],
      max_tokens=300,
      temperature=0,
    )
    plan_text = planner.choices[0].message.content
    print("[planning_agent] response:", plan_text)
    return environment_info, plan_text

  def _load_atomic_actions(self, atomic_actions_path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    actions: List[Dict[str, Any]] = []
    allowed_verbs: List[str] = []

    with open(atomic_actions_path, "r", encoding="utf-8") as f:
      for raw in f:
        line = raw.strip()
        if not line:
          continue
        rec = json.loads(line)
        actions.append(rec)

        action_sig = str(rec.get("action", "")).strip()
        verb = action_sig.split("(", 1)[0].strip()
        if verb == "turn_on":
          verb = "turn on"
        elif verb == "turn_off":
          verb = "turn off"
        elif verb == "break_":
          verb = "break"
        allowed_verbs.append(verb)

    seen = set()
    dedup: List[str] = []
    for v in allowed_verbs:
      if v not in seen:
        seen.add(v)
        dedup.append(v)
    return actions, dedup

  @staticmethod
  def _extract_scene_object_names(objs_from_scene: Optional[List[Dict[str, Any]]]) -> List[str]:
    if not objs_from_scene:
      return []
    names: List[str] = []
    for obj in objs_from_scene:
      if isinstance(obj, dict) and "name" in obj:
        name = str(obj["name"]).strip()
        if name:
          names.append(name)
    return names

  def _sanitize_low_level_plan(
    self,
    steps: List[str],
    allowed_verbs: List[str],
    scene_objects: List[str],
  ) -> List[str]:
    if not steps:
      return []

    allowed_set = set(allowed_verbs)
    scene_lc = [x.casefold() for x in scene_objects]

    def normalize_line(s: str) -> str:
      s = str(s).strip()
      s = re.sub(r"\s+", " ", s)
      s = s.replace("turn_on", "turn on").replace("turn_off", "turn off")
      s = s.replace("move left", "move_left").replace("move right", "move_right")
      s = s.replace("move forward", "move_forward").replace("move back", "move_back")
      s = s.replace("move backward", "move_back")
      return s

    def pick_object_token(payload: str) -> str:
      if not payload:
        return ""
      p = payload.casefold().replace("_", " ").replace("-", " ")
      p = re.sub(r"\s+", " ", p).strip()

      color_words = {
        "red", "blue", "green", "yellow", "orange", "black", "white", "purple", "brown", "gray", "grey"
      }
      toks = [t for t in re.split(r"\s+", p) if t]
      while toks and toks[0] in color_words:
        toks = toks[1:]
      if toks:
        p = " ".join(toks)

      # Normalize common receptacle synonyms before matching.
      synonym_groups = [
        ["basket", "rack", "dish rack", "dish_rack", "dishrack"],
        ["can", "soda", "pepsi", "fanta", "tang"],
      ]
      for group in synonym_groups:
        if any(k in p for k in group):
          for obj in scene_lc:
            if any(k in obj for k in group):
              return obj

      for obj in scene_lc:
        if obj and obj in p:
          return obj

      if any(k in p for k in ["basket", "rack", "dish rack", "dishrack"]):
        return "basket"

      toks = [t for t in re.split(r"\s+", p) if t]
      return toks[-1] if toks else ""

    sanitized: List[str] = []
    current_obj = ""
    for raw in steps:
      s = normalize_line(raw)
      if not s:
        continue

      if s.startswith("turn on"):
        verb = "turn on"
        payload = s[len("turn on") :].strip()
      elif s.startswith("turn off"):
        verb = "turn off"
        payload = s[len("turn off") :].strip()
      else:
        parts = s.split(" ", 1)
        verb = parts[0].strip()
        payload = parts[1].strip() if len(parts) > 1 else ""

      if verb not in allowed_set:
        continue

      if verb in {"move_left", "move_right", "move_forward", "move_back", "drop", "throw", "pour"}:
        if verb in {"move_left", "move_right", "move_forward", "move_back"} and not payload and current_obj:
          payload = current_obj
        sanitized.append(f"{verb} {payload}".strip())
        continue

      obj = pick_object_token(payload)
      if not obj:
        continue
      if verb in {"find", "pick"}:
        current_obj = obj
      sanitized.append(f"{verb} {obj}".strip())

    dedup: List[str] = []
    for step in sanitized:
      if not dedup or dedup[-1] != step:
        dedup.append(step)
    return dedup

  def generate_low_level_plan(
    self,
    high_level_plan: str,
    objs_from_scene: Optional[List[Dict[str, Any]]] = None,
    atomic_actions_path: Optional[str] = None,
  ) -> List[str]:
    if not high_level_plan or not str(high_level_plan).strip():
      return []

    if atomic_actions_path is None:
      atomic_actions_path = str(Path(__file__).with_name("atomic_actions.jsonl"))
    if not os.path.exists(atomic_actions_path):
      raise FileNotFoundError(f"atomic action file not found: {atomic_actions_path}")

    atomic_actions, allowed_verbs = self._load_atomic_actions(atomic_actions_path)
    scene_objects = self._extract_scene_object_names(objs_from_scene)

    prompt = (
      "You are a strict low-level robotic planner.\n"
      "Task: Convert natural-language high-level plan into executable low-level plan.\n"
      "Output format MUST match Python list[str], return JSON array.\n\n"
      "Atomic actions are as follow:\n"
      f"{json.dumps(atomic_actions, ensure_ascii=False)}\n\n"
      "Hard constraints:\n"
      "1) Do not invent verbs, objects, or extra steps not implied by the plan.\n"
      "2) Use find before pick when grasp target is not yet localized.\n"
      "3) Never output invalid actions outside the atomic action (only use and correct use).\n\n"
      "Example:\n"
      "High-level: 'Pick cube, move it to the right, place on table'\n"
      "Output: ['find cube', 'pick cube', 'move_right cube', 'put table']\n\n"
      "High-level: 'Find sphere and drop it'\n"
      "Output: ['find sphere', 'pick sphere', 'drop']\n"
    )

    resp = self.low_level_client.chat.completions.create(
      model=self.low_level_model,
      messages=[
        {"role": "system", "content": "You output strict JSON only."},
        {"role": "user", "content": prompt + "\nHigh-level plan:\n" + str(high_level_plan)},
      ],
      temperature=0,
      max_tokens=400,
      response_format={"type": "json_object"},
    )

    raw: str = (resp.choices[0].message.content or "").strip()
    print("[low_level_planner_llm] raw:", raw)

    steps: List[str] = []
    try:
      parsed = json.loads(raw)
      if isinstance(parsed, dict) and "low_level_plan" in parsed:
        parsed = parsed["low_level_plan"]
      if isinstance(parsed, list):
        steps = [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
      try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
          steps = [str(x).strip() for x in parsed if str(x).strip()]
      except Exception:
        steps = [ln.strip(" -\t") for ln in raw.splitlines() if ln.strip()]

    sanitized = self._sanitize_low_level_plan(steps, allowed_verbs=allowed_verbs, scene_objects=scene_objects)
    print("[low_level_planner_llm] sanitized:", sanitized)
    return sanitized
