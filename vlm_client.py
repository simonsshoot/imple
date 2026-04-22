import ast
import base64
import io
import json
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from openai import OpenAI
from PIL import Image

from config import APIConfig


class LLMClient:
    """Unified client for VLM and text-only LLM calls."""

    def __init__(self, api_config: Optional[APIConfig] = None) -> None:
        self._api_config = api_config or APIConfig()
        self._client_cache: Dict[str, Tuple[OpenAI, str]] = {}

    def _get_client(self, model: str) -> Tuple[OpenAI, str]:
        """Route model name to the correct API endpoint. Cached per model prefix."""
        model_name = (model or "gpt-4o").strip()
        model_lower = model_name.casefold()

        cache_key = model_lower.split("-")[0]
        if cache_key in self._client_cache:
            client, _ = self._client_cache[cache_key]
            return client, model_name

        if model_lower.startswith("deepseek"):
            client = OpenAI(
                api_key=self._api_config.deepseek_api_key,
                base_url=self._api_config.deepseek_base_url,
            )
        elif "qwen" in model_lower or model_lower.startswith("qwq"):
            client = OpenAI(
                api_key=self._api_config.qwen_api_key,
                base_url=self._api_config.qwen_base_url,
            )
        else:
            client = OpenAI(
                api_key=self._api_config.openai_api_key,
                base_url=self._api_config.openai_base_url,
            )

        self._client_cache[cache_key] = (client, model_name)
        return client, model_name

    @staticmethod
    def encode_image(img: np.ndarray, fmt: str = "PNG") -> str:
        """Convert numpy array image to base64 string."""
        arr = np.asarray(img, dtype=np.uint8)
        image = Image.fromarray(arr)
        buf = io.BytesIO()
        image.save(buf, format=fmt)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def chat(
        self,
        model: str,
        messages: List[Dict],
        temperature: float = 0,
        max_tokens: int = 500,
        response_format: Optional[Dict] = None,
    ) -> str:
        """Send a chat completion request and return response text."""
        client, selected_model = self._get_client(model)
        kwargs: Dict[str, Any] = {
            "model": selected_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        resp = client.chat.completions.create(**kwargs)
        return (resp.choices[0].message.content or "").strip()

    def chat_with_image(
        self,
        model: str,
        system_prompt: str,
        user_text: str,
        image: np.ndarray,
        temperature: float = 0,
        max_tokens: int = 500,
    ) -> str:
        """Send a chat completion with an image. Falls back to text-only on failure."""
        encoded = self.encode_image(image)
        try:
            return self.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_text},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{encoded}"},
                            },
                        ],
                    },
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception:
            # Fallback to text-only if image is not supported by the endpoint
            return self.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

    def chat_json(
        self,
        model: str,
        system_prompt: str,
        user_text: str,
        temperature: float = 0,
        max_tokens: int = 400,
    ) -> Any:
        """Send a chat expecting JSON response. Parses with fallback chain."""
        raw = self.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        # Attempt JSON parse, then ast.literal_eval, then line-split
        try:
            return json.loads(raw)
        except Exception:
            pass
        try:
            return ast.literal_eval(raw)
        except Exception:
            pass
        # Return raw lines as list
        return [ln.strip(" -\t") for ln in raw.splitlines() if ln.strip()]

    def chat_with_retry(
        self,
        model: str,
        messages: List[Dict],
        temperature: float = 0,
        max_tokens: int = 500,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> str:
        """Chat with retry on transient errors."""
        last_error: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                return self.chat(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                last_error = exc
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        raise RuntimeError(f"chat failed after {max_retries} retries: {last_error}")
