import argparse
import json
import os
import urllib.error
import urllib.request

import numpy as np

from imple.infos.vlm import DEFAULT_QWEN_BASE_URL, DEFAULT_QWEN_MODEL, QwenVLMClient


def _resolve_api_key(explicit_key: str | None) -> str | None:
	if explicit_key:
		return explicit_key
	return os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")


def _post_json(url: str, payload: dict, api_key: str, timeout_sec: int) -> dict:
	req = urllib.request.Request(
		url=url,
		data=json.dumps(payload).encode("utf-8"),
		headers={
			"Content-Type": "application/json",
			"Authorization": f"Bearer {api_key}",
		},
		method="POST",
	)
	with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
		return json.loads(resp.read().decode("utf-8"))


def test_raw_api(base_url: str, model: str, api_key: str, timeout_sec: int) -> bool:
	"""Call chat/completions directly with text-only input.

	This validates API key + endpoint + model basic response path.
	"""
	payload = {
		"model": model,
		"messages": [
			{"role": "system", "content": "You are a test assistant."},
			{"role": "user", "content": "Reply with EXACTLY: OK"},
		],
		"temperature": 0.0,
		"max_tokens": 32,
		"response_format": {"type": "text"},
	}
	url = f"{base_url.rstrip('/')}/chat/completions"

	try:
		body = _post_json(url=url, payload=payload, api_key=api_key, timeout_sec=timeout_sec)
	except urllib.error.HTTPError as exc:
		msg = exc.read().decode("utf-8", errors="ignore")
		print("[FAIL] Raw API test failed (HTTPError)")
		print(f"  status={exc.code}")
		print(f"  body={msg}")
		return False
	except urllib.error.URLError as exc:
		print("[FAIL] Raw API test failed (URLError)")
		print(f"  error={exc}")
		return False
	except Exception as exc:  # noqa: BLE001
		print("[FAIL] Raw API test failed (Unexpected)")
		print(f"  error={exc}")
		return False

	choices = body.get("choices", [])
	if not choices:
		print("[FAIL] Raw API test failed (no choices)")
		print(f"  response={body}")
		return False

	message = choices[0].get("message", {})
	content = message.get("content", "")
	if isinstance(content, list):
		content = "\n".join(
			item.get("text", "")
			for item in content
			if isinstance(item, dict) and item.get("type") == "text"
		)

	print("[PASS] Raw API test succeeded")
	print(f"  model={body.get('model', model)}")
	print(f"  response_preview={str(content)[:120]}")
	return True


def test_vlm_client(base_url: str, model: str, api_key: str, timeout_sec: int) -> bool:
	"""Call your local QwenVLMClient with a synthetic image.

	This validates your vlm.py wrapper can encode image + call endpoint + parse text.
	"""
	client = QwenVLMClient(
		api_key=api_key,
		model=model,
		base_url=base_url,
		timeout_sec=timeout_sec,
	)

	# Synthetic RGB image for VLM path testing.
	img = np.zeros((64, 64, 3), dtype=np.uint8)
	img[:, :, 1] = 200

	try:
		text = client.chat_with_image(
			image=img,
			system_prompt="You are a test assistant.",
			user_prompt="Reply with one short word: OK",
			response_format={"type": "text"},
			temperature=0.0,
		)
	except ImportError as exc:
		print("[FAIL] VLM client test failed (missing dependency)")
		print(f"  error={exc}")
		print("  hint=Install pillow: pip install pillow")
		return False
	except Exception as exc:  # noqa: BLE001
		print("[FAIL] VLM client test failed")
		print(f"  error={exc}")
		return False

	print("[PASS] VLM client test succeeded")
	print(f"  response_preview={text[:120]}")
	return True


def parse_args():
	parser = argparse.ArgumentParser(
		description="Test Qwen API key and VLM interface responsiveness."
	)
	parser.add_argument(
		"--mode",
		type=str,
		default="all",
		choices=["raw", "vlm", "all"],
		help="Which test(s) to run.",
	)
	parser.add_argument("--api-key", type=str, default="sk-1a68ae54b3fd424f91d0979d7b67b491")
	parser.add_argument("--model", type=str, default=DEFAULT_QWEN_MODEL)
	parser.add_argument("--base-url", type=str, default=DEFAULT_QWEN_BASE_URL)
	parser.add_argument("--timeout-sec", type=int, default=30)
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	api_key = _resolve_api_key(args.api_key)
	if not api_key:
		print("[FAIL] Missing API key")
		print("  set QWEN_API_KEY or DASHSCOPE_API_KEY, or pass --api-key")
		return 2

	ok_raw = True
	ok_vlm = True

	if args.mode in ("raw", "all"):
		ok_raw = test_raw_api(
			base_url=args.base_url,
			model=args.model,
			api_key=api_key,
			timeout_sec=args.timeout_sec,
		)

	if args.mode in ("vlm", "all"):
		ok_vlm = test_vlm_client(
			base_url=args.base_url,
			model=args.model,
			api_key=api_key,
			timeout_sec=args.timeout_sec,
		)

	if ok_raw and ok_vlm:
		print("[DONE] All requested tests passed")
		return 0

	print("[DONE] Some tests failed")
	return 1


if __name__ == "__main__":
	raise SystemExit(main())
