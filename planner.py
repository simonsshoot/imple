"""VLM-based planner for tabletop manipulation tasks.

Refactored from imple/agents.py and imple/run.py into a single
``TabletopPlanner`` class that owns the full  ground -> high-level ->
low-level planning pipeline and supports closed-loop replanning from
visual feedback.
"""

import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from vlm_client import LLMClient
from config import ModelConfig


class TabletopPlanner:
    """End-to-end VLM planner for tabletop robot manipulation."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, llm_client: LLMClient, model_config: ModelConfig) -> None:
        self._client = llm_client
        self._vlm_model = model_config.vlm_model
        self._ll_model = model_config.low_level_planner_model

        # Load the canonical set of atomic actions from the companion file.
        self._atomic_actions, self._allowed_verbs = self._load_atomic_actions()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ground_scene(
        self,
        image: np.ndarray,
        task: str,
        scene_objects: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Use the VLM to describe the scene and identify task-relevant objects.

        Refactored from ``agents.py  sim_ground_agent()``.
        """
        task_objs = self._filter_task_objects(task, scene_objects)

        system_prompt = (
            "You are an assistant that describes the scene and the "
            "task-relevant entities."
        )
        user_text = (
            "Use only objects in this metadata when possible: "
            f"{task_objs}. "
            "Then provide concise task-solving instructions. "
            f"Task: {task}"
        )

        response = self._client.chat_with_image(
            model=self._vlm_model,
            system_prompt=system_prompt,
            user_text=user_text,
            image=image,
            temperature=0,
            max_tokens=500,
        )
        print("[ground_scene] response:", response)
        return response

    def generate_high_level_plan(self, task: str, scene_description: str) -> str:
        """Generate a concise high-level action plan.

        Refactored from the ``planning_agent`` section of
        ``agents.py  multi_agent_vision_planning()``.
        """
        system_prompt = (
            "You produce a concise high-level action plan for robot "
            "manipulation. Output only step lines, no extra commentary."
        )
        user_text = (
            f"Task: {task}\n"
            f"Scene context:\n{scene_description}\n"
            "Return atomic high-level steps."
        )

        response = self._client.chat(
            model=self._vlm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            temperature=0,
            max_tokens=300,
        )
        print("[generate_high_level_plan] response:", response)
        return response

    def generate_low_level_plan(
        self,
        high_level_plan: str,
        scene_objects: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """Convert a high-level plan to executable low-level steps.

        Refactored from ``agents.py  generate_low_level_plan()`` -- uses
        the low-level planner model with JSON output and includes the
        ``_sanitize_steps`` post-processing.
        """
        if not high_level_plan or not str(high_level_plan).strip():
            return []

        obj_names = self._extract_scene_object_names(scene_objects)

        system_prompt = "You output strict JSON only."
        user_text = (
            "You are a strict low-level robotic planner.\n"
            "Task: Convert natural-language high-level plan into "
            "executable low-level plan.\n"
            "Output format MUST match Python list[str], return JSON array.\n\n"
            "Atomic actions are as follow:\n"
            f"{json.dumps(self._atomic_actions, ensure_ascii=False)}\n\n"
            "Hard constraints:\n"
            "1) Do not invent verbs, objects, or extra steps not implied "
            "by the plan.\n"
            "2) Use find before pick when grasp target is not yet "
            "localized.\n"
            "3) Never output invalid actions outside the atomic action "
            "(only use and correct use).\n\n"
            "Example:\n"
            "High-level: 'Pick cube, move it to the right, place on "
            "table'\n"
            "Output: ['find cube', 'pick cube', 'move_right cube', "
            "'put table']\n\n"
            "High-level: 'Find sphere and drop it'\n"
            "Output: ['find sphere', 'pick sphere', 'drop']\n\n"
            f"High-level plan:\n{high_level_plan}"
        )

        parsed = self._client.chat_json(
            model=self._ll_model,
            system_prompt=system_prompt,
            user_text=user_text,
            temperature=0,
            max_tokens=400,
        )

        steps = self._parse_plan_response(parsed)
        print("[generate_low_level_plan] raw steps:", steps)

        sanitized = self._sanitize_steps(steps, obj_names)
        print("[generate_low_level_plan] sanitized:", sanitized)
        return sanitized

    def replan_from_feedback(
        self,
        image: np.ndarray,
        task: str,
        completed_steps: List[str],
        remaining_steps: List[str],
        failure_info: str,
        scene_objects: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """Re-plan remaining steps using the current visual state after an
        execution failure.

        The VLM is shown the current camera image together with context
        about what has been done, what was planned, and why the last step
        failed.  It produces an updated list of remaining low-level steps.
        """
        obj_names = self._extract_scene_object_names(scene_objects)

        system_prompt = (
            "You are a strict low-level robotic replanner. "
            "You are given the current camera observation of the scene, "
            "the original task, steps already executed, steps that were "
            "remaining, and a description of why the most recent step "
            "failed. Your job is to output an updated list of remaining "
            "low-level steps that will complete the task from the current "
            "state.\n\n"
            "Output format MUST be a JSON array of strings, e.g.:\n"
            '[\"find cube\", \"pick cube\", \"move_right cube\", '
            '\"put table\"]\n\n'
            "Hard constraints:\n"
            "1) Only use verbs from the allowed atomic actions.\n"
            "2) Use find before pick when the grasp target is not yet "
            "localized.\n"
            "3) Do not repeat steps that have already been completed "
            "unless the failure requires redoing them.\n"
            "4) Account for the failure information -- if a grasp "
            "failed, you may need to retry find+pick; if placement "
            "failed, you may need to adjust position first.\n\n"
            "Atomic actions are as follow:\n"
            f"{json.dumps(self._atomic_actions, ensure_ascii=False)}"
        )

        user_text = (
            f"ORIGINAL TASK: {task}\n\n"
            f"STEPS COMPLETED SO FAR:\n"
            f"{json.dumps(completed_steps, ensure_ascii=False)}\n\n"
            f"STEPS THAT REMAINED (before failure):\n"
            f"{json.dumps(remaining_steps, ensure_ascii=False)}\n\n"
            f"FAILURE DESCRIPTION:\n{failure_info}\n\n"
            "Given the current camera image of the scene, produce the "
            "updated list of remaining low-level steps (JSON array of "
            "strings) to complete the task from the current state."
        )

        raw_response = self._client.chat_with_image(
            model=self._vlm_model,
            system_prompt=system_prompt,
            user_text=user_text,
            image=image,
            temperature=0,
            max_tokens=500,
        )
        print("[replan_from_feedback] raw response:", raw_response)

        # Parse the response -- try JSON first, then fallback chain.
        steps = self._parse_raw_plan_text(raw_response)
        print("[replan_from_feedback] parsed steps:", steps)

        sanitized = self._sanitize_steps(steps, obj_names)
        print("[replan_from_feedback] sanitized:", sanitized)
        return sanitized

    def plan_full(
        self,
        image: np.ndarray,
        task: str,
        scene_objects: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, str, List[str]]:
        """Full planning pipeline: ground -> high-level -> low-level.

        Returns:
            (scene_description, high_level_plan, low_level_plan)
        """
        scene_desc = self.ground_scene(image, task, scene_objects)
        hl_plan = self.generate_high_level_plan(task, scene_desc)
        ll_plan = self.generate_low_level_plan(hl_plan, scene_objects)
        return scene_desc, hl_plan, ll_plan

    # ------------------------------------------------------------------
    # Static post-processing
    # ------------------------------------------------------------------

    @staticmethod
    def postprocess_plan(scene: str, task: str, plan: List[str]) -> List[str]:
        """Scene-specific post-processing of a low-level plan.

        Refactored from ``imple/run.py  _postprocess_low_level_plan()``.
        Handles PokeCube peg redirect, RollBall fixes, insert
        conversion, and consecutive-duplicate removal.
        """
        if not plan:
            return plan

        scene_l = scene.casefold()
        task_l = task.casefold()
        out = list(plan)

        # -- Insert conversion ----------------------------------------
        # If the task asks for insertion but the plan only has ``put``,
        # convert receptacle-like targets to ``insert``.
        if ("insert" in task_l or "plug" in task_l) and not any(
            s.startswith("insert ") for s in out
        ):
            converted: List[str] = []
            for step in out:
                if step.startswith("put "):
                    target = step[4:].strip().casefold()
                    if any(
                        k in target
                        for k in ["receptacle", "hole", "slot", "socket", "box"]
                    ):
                        converted.append(f"insert {step[4:].strip()}")
                        continue
                converted.append(step)
            out = converted

        # -- PokeCube peg redirect ------------------------------------
        # The PokeCube task uses the peg as tool; when the planner asks
        # to pick the cube, redirect to peg instead.
        if "pokecube" in scene_l:
            redirected: List[str] = []
            for step in out:
                s = step.casefold()
                if s.startswith("find cube"):
                    redirected.append("find peg")
                elif s.startswith("pick cube"):
                    redirected.append("pick peg")
                elif s.startswith("rotate_"):
                    # Avoid unstable rotate in this task; pushing with
                    # peg is sufficient.
                    continue
                else:
                    redirected.append(step)
            out = redirected

        # -- RollBall fixes -------------------------------------------
        # The RollBall scene object is named ``ball`` (not ``sphere``).
        if "rollball" in scene_l:
            normalized: List[str] = []
            has_pick = False
            for step in out:
                s = step.replace("sphere", "ball")
                if s.casefold().startswith("rotate_"):
                    continue
                normalized.append(s)
                if s.casefold().startswith("pick "):
                    has_pick = True
            if not has_pick:
                normalized = ["find ball", "pick ball"] + normalized
            out = normalized

        # -- PartsSorting fixes ----------------------------------------
        # Normalise "tray/container/box" synonyms to "_bin" in put targets.
        if "partssorting" in scene_l:
            fixed: List[str] = []
            for step in out:
                s = step
                for color in ("red", "blue", "green"):
                    for syn in ("tray", "container", "box"):
                        s = s.replace(f"{color} {syn}", f"{color}_bin")
                        s = s.replace(f"{color}_{syn}", f"{color}_bin")
                fixed.append(s)
            out = fixed

        # -- Dedup consecutive duplicates -----------------------------
        deduped: List[str] = []
        for step in out:
            if not deduped or deduped[-1] != step:
                deduped.append(step)
        return deduped

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_atomic_actions(self) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Load ``atomic_actions.jsonl`` from the same directory as this
        module.  Returns ``(actions_list, allowed_verbs)``."""
        actions_path = Path(__file__).with_name("atomic_actions.jsonl")
        if not actions_path.exists():
            raise FileNotFoundError(
                f"atomic action file not found: {actions_path}"
            )

        actions: List[Dict[str, Any]] = []
        allowed_verbs: List[str] = []

        with open(actions_path, "r", encoding="utf-8") as fh:
            for raw in fh:
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

        # Deduplicate while preserving order.
        seen: set = set()
        dedup: List[str] = []
        for v in allowed_verbs:
            if v not in seen:
                seen.add(v)
                dedup.append(v)
        return actions, dedup

    def _sanitize_steps(
        self,
        steps: List[str],
        scene_objects: List[str],
    ) -> List[str]:
        """Validate and normalise steps against allowed verbs and scene
        objects.

        Refactored from ``agents.py  _sanitize_low_level_plan()``.
        """
        if not steps:
            return []

        allowed_set = set(self._allowed_verbs)
        scene_lc = [x.casefold() for x in scene_objects]

        def normalize_line(s: str) -> str:
            s = str(s).strip()
            s = re.sub(r"\s+", " ", s)
            s = s.replace("turn_on", "turn on").replace("turn_off", "turn off")
            s = s.replace("move left", "move_left").replace(
                "move right", "move_right"
            )
            s = s.replace("move forward", "move_forward").replace(
                "move back", "move_back"
            )
            s = s.replace("move backward", "move_back")
            # Normalize push_/pull_/nudge_ variants (e.g. "push_left" -> "push left")
            for verb in ("push", "pull", "nudge"):
                for d in ("left", "right", "forward", "back", "backward"):
                    s = s.replace(f"{verb}_{d}", f"{verb} {d}")
            s = s.replace("backward", "back")
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

            # Extract verb + payload.
            if s.startswith("turn on"):
                verb = "turn on"
                payload = s[len("turn on"):].strip()
            elif s.startswith("turn off"):
                verb = "turn off"
                payload = s[len("turn off"):].strip()
            else:
                parts = s.split(" ", 1)
                verb = parts[0].strip()
                payload = parts[1].strip() if len(parts) > 1 else ""

            if verb not in allowed_set:
                continue

            # Directional moves, drop, throw, pour -- keep payload as-is.
            if verb in {
                "move_left",
                "move_right",
                "move_forward",
                "move_back",
                "drop",
                "throw",
                "pour",
            }:
                if (
                    verb
                    in {"move_left", "move_right", "move_forward", "move_back"}
                    and not payload
                    and current_obj
                ):
                    payload = current_obj
                sanitized.append(f"{verb} {payload}".strip())
                continue

            # push/pull/nudge take "obj direction" -- keep the full payload.
            if verb in {"push", "pull", "nudge"}:
                sanitized.append(f"{verb} {payload}".strip())
                continue

            # Object-bearing verbs -- resolve to a known scene object.
            obj = pick_object_token(payload)
            if not obj:
                continue
            if verb in {"find", "pick"}:
                current_obj = obj
            sanitized.append(f"{verb} {obj}".strip())

        # Remove consecutive duplicates.
        dedup: List[str] = []
        for step in sanitized:
            if not dedup or dedup[-1] != step:
                dedup.append(step)
        return dedup

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_plan_response(parsed: Any) -> List[str]:
        """Extract a ``List[str]`` from the parsed JSON response returned
        by ``LLMClient.chat_json()``.

        ``chat_json`` may return a dict, a list, or a raw-line fallback
        list.  This helper normalises all cases.
        """
        if isinstance(parsed, dict):
            # Try common wrapper keys.
            for key in ("low_level_plan", "plan", "steps", "actions"):
                if key in parsed:
                    parsed = parsed[key]
                    break
            else:
                # Take the first list-valued entry.
                for v in parsed.values():
                    if isinstance(v, list):
                        parsed = v
                        break
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
        return []

    @staticmethod
    def _parse_raw_plan_text(raw: str) -> List[str]:
        """Best-effort extraction of a step list from free-form VLM text.

        Tries JSON parsing, then ``ast.literal_eval``, then falls back
        to line splitting.
        """
        raw = raw.strip()

        # Attempt to extract a JSON array embedded in the text.
        # The VLM sometimes wraps the array in markdown fences or extra
        # commentary.
        json_match = re.search(r"\[.*\]", raw, re.DOTALL)
        candidate = json_match.group(0) if json_match else raw

        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass

        try:
            parsed = ast.literal_eval(candidate)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass

        # Fallback: split on newlines.
        return [
            ln.strip(" -\t")
            for ln in raw.splitlines()
            if ln.strip() and not ln.strip().startswith("```")
        ]

    # ------------------------------------------------------------------
    # Object filtering
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_task_objects(
        task: str,
        objs: Optional[List[Dict[str, Any]]],
    ) -> Optional[List[Dict[str, Any]]]:
        """Filter scene object metadata to only task-relevant items.

        Refactored from the ``filter_task_objects`` closure inside
        ``agents.py  multi_agent_vision_planning()``.
        """
        if not objs:
            return objs

        task_words = set(task.lower().replace("-", " ").split())
        always_keep = {
            "sphere",
            "ball",
            "cube",
            "cylinder",
            "table",
            "workspace",
            "robot",
            "panda",
            "arm",
            "bin",
            "tray",
            "container",
        }

        filtered: List[Dict[str, Any]] = []
        for obj in objs:
            if not isinstance(obj, dict):
                continue
            name = str(obj.get("name", ""))
            tokens = set(name.lower().replace("_", " ").replace("-", " ").split())
            if (tokens & task_words) or (tokens & always_keep):
                filtered.append(obj)

        return filtered if filtered else objs

    @staticmethod
    def _extract_scene_object_names(
        objs: Optional[List[Dict[str, Any]]],
    ) -> List[str]:
        """Return a flat list of object name strings from scene metadata."""
        if not objs:
            return []
        names: List[str] = []
        for obj in objs:
            if isinstance(obj, dict) and "name" in obj:
                name = str(obj["name"]).strip()
                if name:
                    names.append(name)
        return names
