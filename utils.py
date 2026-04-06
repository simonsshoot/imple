import base64
import io
import os
import re
import traceback
from typing import Iterable, List

import numpy as np
from openai import OpenAI
from PIL import Image


# Default endpoints/keys for different providers.
OPENAI_API_KEY = "sk-zk280d44e707a4a809a4c467266a213db66693bf03745f72"
OPENAI_BASE_URL = "https://api.zhizengzeng.com/v1"

DEEPSEEK_API_KEY = "sk-31191053cbb54460a55393173ec3892a"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWEN_MODEL = "qwen-vl-max"
API_KEY = "sk-1a68ae54b3fd424f91d0979d7b67b491"


def model_selection(model:str="gpt-4o"):
  """根据args参数返回模型的client"""
  model_name = (model or "gpt-4o").strip()
  model_lower = model_name.casefold()

  if model_lower.startswith("deepseek"):
    api_key = os.getenv("DEEPSEEK_API_KEY", DEEPSEEK_API_KEY)
    base_url = os.getenv("DEEPSEEK_BASE_URL", DEEPSEEK_BASE_URL)
    client = OpenAI(api_key=api_key, base_url=base_url)
    return client, model_name

  if model_lower.startswith("qwen") or model_lower.startswith("qwq") or "qwen" in model_lower:
    api_key = os.getenv("QWEN_API_KEY", os.getenv("DASHSCOPE_API_KEY", API_KEY))
    base_url = os.getenv("QWEN_BASE_URL", DEFAULT_QWEN_BASE_URL)
    selected_model = model_name if model_name else DEFAULT_QWEN_MODEL
    client = OpenAI(api_key=api_key, base_url=base_url)
    return client, selected_model

  # Default route: OpenAI-compatible endpoint.
  api_key = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)
  base_url = os.getenv("OPENAI_BASE_URL", OPENAI_BASE_URL)
  client = OpenAI(api_key=api_key, base_url=base_url)
  return client, model_name


def ndarray_to_base64(img_array: np.ndarray, image_format: str = "PNG") -> str:
  """
        将 numpy.ndarray 格式的图像转换为 Base64 编码格式
        返回:
        str: 纯 Base64 编码的图像字符串，不包含前缀。
  """
  img_array = img_array.astype(np.uint8)
  image = Image.fromarray(img_array)
  buffered = io.BytesIO()
  image.save(buffered, format=image_format)
  img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
  return img_str

def save_pics(img, file_path: str, image_format: str = "PNG"):
  arr = np.asarray(img)

  # Normalize common ManiSkill image layouts to HxWxC.
  if arr.ndim == 4:
    # Typical cases: [N, H, W, C] or [1, H, W, C]. Save the first frame.
    arr = arr[0]
  if arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[-1] != 1:
    # Rare case from wrappers: [1, H, W] or [1, H, W, C] squeezed above.
    arr = np.squeeze(arr, axis=0)
  if arr.ndim == 2:
    # Grayscale -> PIL accepts HxW directly.
    pass
  elif arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
    pass
  else:
    raise TypeError(f"Unsupported image shape for save_pics: {arr.shape}")

  Image.fromarray(arr.astype(np.uint8)).save(file_path, format=image_format)


def gen_low_level_plan(high_level_plan: str):
  """高层计划到低层计划的转换器,将自然语言描述的高层任务计划转换为机器人可以直接执行的底层动作序列。"""
  if not high_level_plan:
    return []

  # 允许输入是字符串(多行)或列表(每行一个步骤)。
  if isinstance(high_level_plan, str):
    lines = [line.strip(" -\t") for line in high_level_plan.splitlines() if line.strip()]
  elif isinstance(high_level_plan, Iterable):
    lines = [str(line).strip(" -\t") for line in high_level_plan if str(line).strip()]
  else:
    return []

  # 标准低层动作集合（需与 Controller.llm_skill_interact 保持一致）。
  valid_prefixes = [
    "find",
    "pick",
    "put",
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

  # 将常见自然语言动词映射到标准低层动词。
  keyword_to_action = [
    (r"\b(find|go to|walk to|move to|approach|locate|search|face)\b", "find"),
    (r"\b(pick up|pickup|pick|grab|grasp|take|collect)\b", "pick"),
    (r"\b(place|put|insert|drop into|put down on|set on)\b", "put"),
    (r"\b(open)\b", "open"),
    (r"\b(close|shut)\b", "close"),
    (r"\b(slice|cut|chop)\b", "slice"),
    (r"\b(turn on|switch on|toggle on|enable|start)\b", "turn on"),
    (r"\b(turn off|switch off|toggle off|disable|stop)\b", "turn off"),
    (r"\b(throw|toss)\b", "throw"),
    (r"\b(drop|release|let go)\b", "drop"),
    (r"\b(break|smash)\b", "break"),
    (r"\b(cook|heat)\b", "cook"),
    (r"\b(dirty|soil)\b", "dirty"),
    (r"\b(clean|wash)\b", "clean"),
    (r"\b(fill liquid|fill)\b", "fillLiquid"),
    (r"\b(empty liquid|empty)\b", "emptyLiquid"),
    (r"\b(pour)\b", "pour"),
  ]

  stop_words = {
    "the",
    "a",
    "an",
    "to",
    "into",
    "in",
    "on",
    "at",
    "from",
    "of",
    "and",
    "then",
    "finally",
    "next",
    "towards",
    "toward",
    "with",
  }

  object_vocab = {
    "cube",
    "sphere",
    "ball",
    "box",
    "bin",
    "mug",
    "cup",
    "bottle",
    "table",
    "surface",
    "object",
  }

  def _clean_text(text: str) -> str:
    text = text.strip().strip(".,;:")
    text = re.sub(r"^\d+[\).]\s*", "", text)
    return text

  def _extract_object_phrase(text: str, action: str) -> str:
    text_l = text.casefold()
    # 去掉动词前缀。
    text_l = re.sub(r"^(please\s+)?(now\s+)?", "", text_l)
    text_l = re.sub(r"^(turn\s+around\s+and\s+)?", "", text_l)
    text_l = re.sub(r"^(walk\s+to\s+and\s+)?", "", text_l)

    if action == "turn on":
      text_l = re.sub(r"^.*?(turn\s+on|switch\s+on|toggle\s+on)\s+", "", text_l)
    elif action == "turn off":
      text_l = re.sub(r"^.*?(turn\s+off|switch\s+off|toggle\s+off)\s+", "", text_l)
    elif action == "fillLiquid":
      text_l = re.sub(r"^.*?fill\s+", "", text_l)
    elif action == "emptyLiquid":
      text_l = re.sub(r"^.*?empty\s+", "", text_l)
    else:
      text_l = re.sub(r"^[a-z\s]*?\b" + re.escape(action).replace("\\ ", "\\s+") + r"\b\s*", "", text_l)

    # 若句子包含介词短语，优先取介词后的客体（put X in microwave -> microwave）。
    prep_match = re.search(r"\b(in|into|on|onto|to)\s+([a-z0-9_\-\s]+)$", text_l)
    if prep_match and action in {"put", "find"}:
      candidate = prep_match.group(2)
    else:
      candidate = text_l

    tokens = [t for t in re.split(r"\s+", candidate.strip()) if t and t not in stop_words]
    if not tokens:
      return ""

    # 控制目标短语长度，避免整句都进来。
    phrase = " ".join(tokens[:4]).strip(".,;:")
    return phrase

  def _extract_salient_object(text: str) -> str:
    text_l = text.casefold()
    # Keep short object phrases like "red cube", "wooden surface", "the bin".
    candidates = re.findall(
      r"\b(?:the\s+|a\s+|an\s+)?(?:red|blue|green|yellow|black|white|wooden|small|large\s+)?"
      r"(cube|sphere|ball|box|bin|mug|cup|bottle|table|surface|object)\b",
      text_l,
    )
    if not candidates:
      return ""
    return candidates[-1]

  def _normalize_obj_phrase(obj_phrase: str) -> str:
    p = obj_phrase.casefold().strip()
    # Map generic language to scene-friendly names.
    p = re.sub(r"\b(ball|orb)\b", "sphere", p)
    p = re.sub(r"\b(surface)\b", "table", p)
    p = re.sub(r"\b(wooden table|wood table)\b", "table", p)
    # Remove manipulator words that are not scene entities.
    p = re.sub(r"\b(robot|arm|gripper|hand|end effector|end-effector)\b", "", p)
    p = re.sub(r"\s+", " ", p).strip()
    return p

  low_level_plan: List[str] = []
  for raw_line in lines:
    line = _clean_text(raw_line)
    if not line:
      continue

    line_l = line.casefold()

    # Skip pure motion-only lines (they are usually transitional and not object-level skills).
    if re.search(r"\b(move|raise|lower|lift|shift)\b", line_l) and re.search(r"\b(left|right|up|down|downward|upward)\b", line_l):
      if not re.search(r"\b(grasp|pick|grab|place|put|release|drop)\b", line_l):
        continue

    # Special handling for gripper-language plans.
    if re.search(r"\bclose\b", line_l) and re.search(r"\bgripper\b", line_l) and re.search(r"\b(pick|grasp|grab|take)\b", line_l):
      obj = _normalize_obj_phrase(_extract_salient_object(line_l))
      low_level_plan.append(f"pick {obj}" if obj else "pick")
      continue
    if re.search(r"\bopen\b", line_l) and re.search(r"\bgripper\b", line_l) and re.search(r"\b(release|drop|let go)\b", line_l):
      low_level_plan.append("drop")
      continue

    if re.search(r"\b(place|put)\b", line_l) and re.search(r"\b(surface|table)\b", line_l):
      low_level_plan.append("put table")
      continue

    # 若本身已是标准低层动作，直接接收。
    matched_standard = False
    for prefix in valid_prefixes:
      if line_l.startswith(prefix.casefold()):
        low_level_plan.append(line_l)
        matched_standard = True
        break
    if matched_standard:
      continue

    # 否则尝试用规则映射到标准动作。
    mapped_action = None
    for pat, act in keyword_to_action:
      if re.search(pat, line_l):
        mapped_action = act
        break

    if mapped_action is None:
      # 无法转换时跳过该行（稳健性优先）。
      continue

    if mapped_action in {"drop", "throw", "pour"}:
      low_level_plan.append(mapped_action)
      continue

    # Handle common motion-language outputs from planning models.
    if mapped_action == "pick":
      if "sphere" in line_l:
        low_level_plan.append("pick sphere")
        continue
      if "ball" in line_l:
        low_level_plan.append("pick ball")
        continue

    if mapped_action == "put":
      if "bin" in line_l:
        low_level_plan.append("put bin")
        continue
      if "box" in line_l:
        low_level_plan.append("put box")
        continue

    if mapped_action == "find":
      if "sphere" in line_l:
        low_level_plan.append("find sphere")
        continue
      if "ball" in line_l:
        low_level_plan.append("find ball")
        continue
      if "table" in line_l:
        low_level_plan.append("find table")
        continue

    obj_phrase = _extract_object_phrase(line, mapped_action)
    if obj_phrase:
      obj_phrase = _normalize_obj_phrase(obj_phrase)

    # If extracted phrase is obviously not an entity, fallback to salient object noun.
    if (
      not obj_phrase
      or re.search(r"\b(robot|arm|gripper|downward|upward|left|right|move)\b", obj_phrase)
      or obj_phrase.split()[-1] not in object_vocab
    ):
      salient = _normalize_obj_phrase(_extract_salient_object(line))
      if salient:
        obj_phrase = salient

    if not obj_phrase:
      # 兜底：没有客体时仍保留动词，交给 controller 给出失败信息。
      low_level_plan.append(mapped_action)
    else:
      if mapped_action == "fillLiquid":
        # fill 需要液体名，默认 water。
        if obj_phrase.split()[-1] not in {"water", "coffee", "wine"}:
          obj_phrase = f"{obj_phrase} water"
      low_level_plan.append(f"{mapped_action} {obj_phrase}".strip())

  # Remove duplicates while preserving order.
  dedup_plan: List[str] = []
  for step in low_level_plan:
    if len(dedup_plan) == 0 or dedup_plan[-1] != step:
      dedup_plan.append(step)
  low_level_plan = dedup_plan

  # Canonicalize common object synonyms for ManiSkill tasks.
  canon_map = {
    "ball": "sphere",
    "orb": "sphere",
  }
  canonicalized: List[str] = []
  for step in low_level_plan:
    s = step
    for src, dst in canon_map.items():
      s = re.sub(rf"\b{src}\b", dst, s)
    canonicalized.append(s)
  low_level_plan = canonicalized

  # If we try to pick an object without a prior find, prepend a find step.
  improved_plan: List[str] = []
  found_targets = set()
  for step in low_level_plan:
    if step.startswith("find "):
      found_targets.add(step.replace("find ", "", 1).strip())
      improved_plan.append(step)
      continue

    if step.startswith("pick "):
      target = step.replace("pick ", "", 1).strip()
      if target and target not in found_targets:
        improved_plan.append(f"find {target}")
        found_targets.add(target)
      improved_plan.append(step)
      continue

    improved_plan.append(step)
  low_level_plan = improved_plan

  # Task-specific fallback: ensure non-empty executable plan.
  if len(low_level_plan) == 0:
    print("fallback to default plan")
    text = "\n".join(lines).lower()
    obj_name = "sphere" if "sphere" in text or "ball" in text else "object"
    if "bin" in text:
      low_level_plan = [f"find {obj_name}", f"pick {obj_name}", "find bin", "put bin"]
    elif "box" in text:
      low_level_plan = [f"find {obj_name}", f"pick {obj_name}", "find box", "put box"]
    else:
      low_level_plan = [f"find {obj_name}", f"pick {obj_name}", "drop"]
  print(f"[gen_low_level_plan] high_level_plan:\n{high_level_plan}\n=> low_level_plan:\n{low_level_plan}")
  return low_level_plan

def execute_low_level_plan(planner, low_level_plan):  
  """执行底层计划的函数,接收一个低层计划和一个控制器实例，解析计划中的动作指令，并调用控制器的方法来执行这些动作。"""
  if not hasattr(planner, "llm_skill_interact"):
    raise AttributeError("planner must provide llm_skill_interact(instruction)")

  if low_level_plan is None:
    low_level_plan = []

  if isinstance(low_level_plan, str):
    steps = [line.strip() for line in low_level_plan.splitlines() if line.strip()]
  else:
    steps = [str(x).strip() for x in low_level_plan if str(x).strip()]

  if hasattr(planner, "restore_scene"):
    planner.restore_scene()

  logs = []
  num_success_steps = 0

  for i, step in enumerate(steps, start=1):
    try:
      ret_dict = planner.llm_skill_interact(step)
      success = bool(ret_dict.get("success", False))
      if success:
        num_success_steps += 1

      logs.append(
        {
          "step_idx": i,
          "instruction": step,
          "success": success,
          "message": ret_dict.get("message", ""),
          "errorMessage": ret_dict.get("errorMessage", ""),
        }
      )
      print(ret_dict)
      print("-" * 50)
    except Exception as exc:
      traceback.print_exc()
      logs.append(
        {
          "step_idx": i,
          "instruction": step,
          "success": False,
          "message": f"Exception: {exc}",
          "errorMessage": str(exc),
        }
      )

  sr_step = 0.0 if len(steps) == 0 else num_success_steps / len(steps)
  result = {
    "num_total_steps": len(steps),
    "num_success_steps": num_success_steps,
    "success_rate": sr_step,
    "logs": logs,
  }
  print(f"[execute_low_level_plan] result: {result}")
  return result