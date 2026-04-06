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

    def environment_agent():
      """提取其中的主要物体及其空间关系，并以(主体, 关系, 客体)的三元组格式返回描述。"""
      agent = self.client.chat.completions.create(
        model=self.model,
        messages=[
          {
            "role": "user",
            "content": [
              {
                "type": "text",
                "text": "You are an assistent which is able accurately describe the content of an image. \n\
                        In particular, you are able to capture the main objects present.\n\
                        Explore the image accurately as an expert and find all the object that you can see.\n\
                        in the image and provide the relations that exist between them. \n\
                        These relations are described in the form of a triple (subject, relation, object) \
                        and when you answer you are only expected to answer with triples and nothing else. \n\
                        When writing the triples, try to execute this task: " + self.task_description + "\n\
                        and verify the elements that you neeed to solve and write the relation of the objects in the image.\n\
                        For example, if in a scene there is a door, a table in front of the door and a book on the table \
                        with a pen right to it, your answer should be: \
                        1) (table, in front of, door) \n\
                        2) (book, on, table) \n\
                        3) (pen, on, table) \n\
                        4) (pen, right to, book) \n\
                        5) (book, left to, pen). \n\
                        Only mention objects that are necessary for the current task.\n\
                        Do not introduce irrelevant objects even if they exist in scene metadata.\n\
                        At the end of the task, you must write a instruction to solve the task, in a way that you can\
                        help who read your answer to understand how to solve the task without knowing the scene.",
              },
              {
                "type": "image_url",
                "image_url": {
                  "url": f"data:image/jpeg;base64,{self.encoded_image}",
                },
              },
            ],
          }
        ],
        max_tokens=500,
        temperature=0,
      )

      response = agent.choices[0].message.content
      print("[environment_agent] response:", response)
      return response

    def sim_ground_agent(objs_from_scene=None):
      """在给定场景物体先验知识（objs_from_scene）的约束下，生成一份详尽、无歧义的自然语言场景描述。"""
      prompt = (
        f"Here is the oracle objects involved in the task: \n{task_objs}\n"
        "Do not use any objects not in the scene. "
        "Only mention objects strictly relevant to the task goal."
      )
      agent = self.client.chat.completions.create(
        model=self.model,
        messages=[
          {
            "role": "user",
            "content": [
              {
                "type": "text",
                "text": "You are an assistent which is able accurately describe the content of an image. \n\
                            In particular, you are able to describe accurately the content of the image to make one understand \
                            all the details of the image without seeing it. \n\
                            You should describe how the scene it is made with high level description and precise instruction to solve\
                            the following task : " + self.task_description + prompt + "\n\
                            If the task contains ambiguity in the solution of the task , for example same objects of the same type,\
                            specify the position of the object in the image or in relation at other objects.\n",
              },
              {
                "type": "image_url",
                "image_url": {
                  "url": f"data:image/jpeg;base64,{self.encoded_image}",
                },
              },
            ],
          }
        ],
        max_tokens=500,
        temperature=0,
      )

      response = agent.choices[0].message.content
      print("[sim_ground_agent] response:", response)
      return response

    enviroment_info = environment_agent() + "\n" + sim_ground_agent(task_objs)
    agent = self.client.chat.completions.create(
      model=self.model,
      messages=[
        {
          "role": "system",
          "content": "You are an  helpful assistant which is able accurately describe the navigation planning step to reach the required goal.\n\
             You know how are the object that you can use and where are from the following information "
          + enviroment_info
          + "\
             You will do a planning to execute the goal using the information written.\n\
            Use only task-relevant objects and avoid unrelated distractors.\n\
            Your answer will be a list of only steps that help a agent to reach the goal. Try to do precise information for each step but in atomic way\n\
            Your answer will be as that in the following example adding the navigation operation (Turn , move ,walk)\
                and containing only the atomic step with the position of the object and nothing else.\n\
                For example if the goal is 'Place a heated glass in a cabinet' your answer using the objects \
                    perceived in the enviroment will be: \n\
                   Turn around and walk to the sink.,\n\
                   Take the left glass out of the sink.,\n\
                    Turn around and walk to the microwave.,\n\
                    Heat the glass in the microwave.,\n\
                    Turn around and face the counter.,\n\
                    Place the glass in the left top cabinet.\n",
        },
        {"role": "user", "content": "The goal is " + self.task_description},
      ],
      temperature=0,
      max_tokens=300,
    )
    print("[planning_agent] response:", agent.choices[0].message.content)
    return enviroment_info, agent.choices[0].message.content

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
      p = payload.casefold()
      for obj in scene_lc:
        if obj and obj in p:
          return obj
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
