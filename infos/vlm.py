import argparse
import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional

import gymnasium as gym
import mani_skill.envs
import numpy as np


DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWEN_MODEL = "qwen-vl-max"
API_KEY = "sk-1a68ae54b3fd424f91d0979d7b67b491"


def _extract_first_json_object(text: str) -> Dict[str, Any]:
	"""Extract and parse the first JSON object from model text output."""
	text = text.strip()
	try:
		return json.loads(text)
	except json.JSONDecodeError:
		pass

	match = re.search(r"\{.*\}", text, flags=re.DOTALL)
	if not match:
		raise ValueError(f"No JSON object found in model output: {text}")
	return json.loads(match.group(0))


def _image_to_data_url(image: np.ndarray, fmt: str = "png") -> str:
	"""Convert an RGB image ndarray to a base64 data URL.

	We use PNG because it is lossless and widely accepted by VLM APIs.
	"""
	from io import BytesIO

	try:
		from PIL import Image
	except ImportError as exc:
		raise ImportError(
			"Pillow is required for image encoding. Install with: pip install pillow"
		) from exc

	if image.dtype != np.uint8:
		image = np.clip(image, 0, 255).astype(np.uint8)
	if image.ndim != 3 or image.shape[-1] != 3:
		raise ValueError(f"Expected RGB image with shape [H, W, 3], got {image.shape}")

	im = Image.fromarray(image)
	buf = BytesIO()
	im.save(buf, format=fmt.upper())
	payload = base64.b64encode(buf.getvalue()).decode("ascii")
	return f"data:image/{fmt};base64,{payload}"


@dataclass
class VLMAction:
	"""Action for Panda + pd_ee_delta_pose.

	dx, dy, dz: end-effector translation delta
	droll, dpitch, dyaw: end-effector rotation delta in rad
	gripper: [-1, 1], negative closes gripper, positive opens gripper
	"""

	dx: float
	dy: float
	dz: float
	droll: float
	dpitch: float
	dyaw: float
	gripper: float

	def to_numpy(self, clip: bool = True) -> np.ndarray:
		arr = np.array(
			[
				self.dx,
				self.dy,
				self.dz,
				self.droll,
				self.dpitch,
				self.dyaw,
				self.gripper,
			],
			dtype=np.float32,
		)
		if clip:
			arr[:6] = np.clip(arr[:6], -1.0, 1.0)
			arr[6] = float(np.clip(arr[6], -1.0, 1.0))
		return arr


class QwenVLMClient:
	"""Minimal OpenAI-compatible client for Qwen series VLM models."""

	def __init__(
		self,
		api_key: Optional[str] = None,
		model: str = DEFAULT_QWEN_MODEL,
		base_url: str = DEFAULT_QWEN_BASE_URL,
		timeout_sec: int = 30,
	):
		self.api_key = API_KEY
		if not self.api_key:
			raise ValueError(
				"Missing API key. Set QWEN_API_KEY or DASHSCOPE_API_KEY in environment."
			)
		self.model = model
		self.base_url = base_url.rstrip("/")
		self.timeout_sec = timeout_sec

	def chat_with_image(
		self,
		image: np.ndarray,
		user_prompt: str,
		system_prompt: str,
		response_format: Optional[dict] = None,
		temperature: float = 0.0,
	) -> str:
		data_url = _image_to_data_url(image, fmt="png")
		payload = {
			"model": self.model,
			"messages": [
				{"role": "system", "content": system_prompt},
				{
					"role": "user",
					"content": [
						{"type": "text", "text": user_prompt},
						{"type": "image_url", "image_url": {"url": data_url}},
					],
				},
			],
			"temperature": temperature,
		}
		if response_format is not None:
			payload["response_format"] = response_format
		req = urllib.request.Request(
			url=f"{self.base_url}/chat/completions",
			data=json.dumps(payload).encode("utf-8"),
			headers={
				"Content-Type": "application/json",
				"Authorization": f"Bearer {self.api_key}",
			},
			method="POST",
		)
		try:
			with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
				body = json.loads(resp.read().decode("utf-8"))
		except urllib.error.HTTPError as exc:
			msg = exc.read().decode("utf-8", errors="ignore")
			raise RuntimeError(f"Qwen API HTTPError {exc.code}: {msg}") from exc
		except urllib.error.URLError as exc:
			raise RuntimeError(f"Qwen API URLError: {exc}") from exc

		choices = body.get("choices", [])
		if not choices:
			raise RuntimeError(f"No choices in API response: {body}")
		message = choices[0].get("message", {})
		content = message.get("content", "")
		if isinstance(content, list):
			parts = []
			for item in content:
				if isinstance(item, dict) and item.get("type") == "text":
					parts.append(item.get("text", ""))
			content = "\n".join(parts)
		if not isinstance(content, str) or not content.strip():
			raise RuntimeError(f"Empty text content in API response: {body}")
		return content


class ManiSkillVLMPolicy:
	"""VLM policy that maps image + task text to a safe 7D action.

	Output action assumes `control_mode='pd_ee_delta_pose'`.
	"""
	ACTION_SCHEMA = {
		"dx": "float in [-1,1]",
		"dy": "float in [-1,1]",
		"dz": "float in [-1,1]",
		"droll": "float in [-1,1]",
		"dpitch": "float in [-1,1]",
		"dyaw": "float in [-1,1]",
		"gripper": "float in [-1,1], negative=close, positive=open",
		"reason": "short explanation string",
	}
	ACTION_JSON_SCHEMA = {
		"name": "maniskill_single_step_action",
		"strict": True,
		"schema": {
			"type": "object",
			"properties": {
				"dx": {"type": "number", "minimum": -1.0, "maximum": 1.0},
				"dy": {"type": "number", "minimum": -1.0, "maximum": 1.0},
				"dz": {"type": "number", "minimum": -1.0, "maximum": 1.0},
				"droll": {"type": "number", "minimum": -1.0, "maximum": 1.0},
				"dpitch": {"type": "number", "minimum": -1.0, "maximum": 1.0},
				"dyaw": {"type": "number", "minimum": -1.0, "maximum": 1.0},
				"gripper": {"type": "number", "minimum": -1.0, "maximum": 1.0},
				"reason": {"type": "string"},
			},
			"required": [
				"dx",
				"dy",
				"dz",
				"droll",
				"dpitch",
				"dyaw",
				"gripper",
				"reason",
			],
			"additionalProperties": False,
		},
	}

	def __init__(self, vlm_client: QwenVLMClient):
		self.vlm_client = vlm_client

	def _build_system_prompt(self) -> str:
		return (
			"You are a robot policy for ManiSkill. "
			"Given one RGB frame and a task goal, output exactly one JSON object for the NEXT env.step only. "
			"Do not plan multiple steps. Do not output markdown or extra text. Keep motion conservative and safe. "
			"Action semantics: dx/dy/dz are one-step EE translation deltas, droll/dpitch/dyaw are one-step EE rotation deltas, "
			"gripper < 0 means close and gripper > 0 means open. "
			f"Schema: {json.dumps(self.ACTION_SCHEMA, ensure_ascii=True)}"
		)

	def _build_user_prompt(self, task_prompt: str, step_idx: int) -> str:
		return (
			f"Task: {task_prompt}\n"
			f"Current step: {step_idx}\n"
			"Return one immediate action JSON for the next env.step now."
		)

	def _response_format_json_schema(self) -> dict:
		return {
			"type": "json_schema",
			"json_schema": self.ACTION_JSON_SCHEMA,
		}

	def _safe_parse_action(self, model_text: str) -> VLMAction:
		obj = _extract_first_json_object(model_text)
		return VLMAction(
			dx=float(obj.get("dx", 0.0)),
			dy=float(obj.get("dy", 0.0)),
			dz=float(obj.get("dz", 0.0)),
			droll=float(obj.get("droll", 0.0)),
			dpitch=float(obj.get("dpitch", 0.0)),
			dyaw=float(obj.get("dyaw", 0.0)),
			gripper=float(obj.get("gripper", 1.0)),
		)

	def act(self, image: np.ndarray, task_prompt: str, step_idx: int) -> np.ndarray:
		system_prompt = self._build_system_prompt()
		user_prompt = self._build_user_prompt(task_prompt=task_prompt, step_idx=step_idx)
		try:
			raw = self.vlm_client.chat_with_image(
				image=image,
				user_prompt=user_prompt,
				system_prompt=system_prompt,
				response_format=self._response_format_json_schema(),
				temperature=0.0,
			)
		except RuntimeError as exc:
			# Some endpoints/models may not fully support json_schema yet; fallback to json_object.
			if "response_format" not in str(exc):
				raise
			raw = self.vlm_client.chat_with_image(
				image=image,
				user_prompt=user_prompt,
				system_prompt=system_prompt,
				response_format={"type": "json_object"},
				temperature=0.0,
			)
		action = self._safe_parse_action(raw).to_numpy(clip=True)
		return action


def run_vlm_episode(
	env_id: str,
	task_prompt: str,
	max_steps: int,
	video_dir: str,
	model: str,
	base_url: str,
	seed: int,
):
	"""Run one ManiSkill episode driven by Qwen-VL policy."""
	from mani_skill.utils.wrappers.record import RecordEpisode

	env = gym.make(
		env_id,
		obs_mode="state",
		control_mode="pd_ee_delta_pose",
		render_mode="rgb_array",
		render_backend="cpu",
		sim_backend="cpu",
	)

	env = RecordEpisode(
		env,
		output_dir=video_dir,
		save_trajectory=False,
		save_video=True,
		video_fps=20,
		source_type="vlm",
		source_desc=f"Qwen policy, model={model}",
	)

	client = QwenVLMClient(model=model, base_url=base_url)
	policy = ManiSkillVLMPolicy(client)

	_, _ = env.reset(seed=seed)
	terminated = False
	truncated = False

	for step_idx in range(max_steps):
		frame = env.render()
		frame = np.asarray(frame)
		if frame.ndim == 4:
			frame = frame[0]

		try:
			print(f"Step {step_idx}: querying VLM policy...")
			action = policy.act(frame, task_prompt=task_prompt, step_idx=step_idx)
			print(f"VLM action: {action}")
		except Exception as exc:
			# Safe fallback: keep still and open gripper so the episode can continue.
			print(f"[WARN] VLM act failed at step {step_idx}: {exc}")
			action = np.zeros(7, dtype=np.float32)
			action[6] = 1.0

		_, _, terminated, truncated, _ = env.step(action)
		if bool(terminated) or bool(truncated):
			break
		# Small delay is useful when debugging API quota/rate limits.
		time.sleep(0.02)

	env.flush_video(name="vlm_episode")
	env.close()
	print(f"Finished. terminated={terminated}, truncated={truncated}")


def parse_args():
	parser = argparse.ArgumentParser(description="Use Qwen-VL to control ManiSkill.")
	parser.add_argument("--env-id", type=str, default="PlaceSphere-v1")
	parser.add_argument(
		"--task-prompt",
		type=str,
		default="Move the sphere from left to right and place it near the bin.",
	)
	parser.add_argument("--max-steps", type=int, default=120)
	parser.add_argument("--video-dir", type=str, default="videos/place_sphere")
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--model", type=str, default=DEFAULT_QWEN_MODEL)
	parser.add_argument("--base-url", type=str, default=DEFAULT_QWEN_BASE_URL)
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	run_vlm_episode(
		env_id=args.env_id,
		task_prompt=args.task_prompt,
		max_steps=args.max_steps,
		video_dir=args.video_dir,
		model=args.model,
		base_url=args.base_url,
		seed=args.seed,
	)
