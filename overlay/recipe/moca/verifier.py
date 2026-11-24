"""Small, bounded OpenAI-compatible client for a frozen local text verifier."""

import asyncio
import json
import math
import re

import aiohttp
from recipe.moca.prompts import PV_PROMPT, SVV_PROMPT


def parse_svv_score(text: str) -> float:
    values = re.findall(r"\\boxed\{\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\}", text)
    if len(values) == 1 and len(re.findall(r"\\boxed\s*\{", text)) == 1:
        value = values[0]
    else:
        # Some local judges quote the student's boxed answer despite the prompt.
        # In that case only accept a distinct, standalone terminal score line.
        terminal = re.search(r"(?:^|\n)\s*\\boxed\{\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\}\s*$", text)
        if terminal is None:
            raise ValueError("SVV must return an unambiguous final numeric boxed score")
        value = terminal[1]
    score = float(value)
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("SVV score outside [0, 1]")
    return score


class VerifierClient:
    def __init__(self, config):
        self.config = config
        self.semaphore = asyncio.Semaphore(int(config.get("max_concurrency", 1)))

    async def _chat(self, system: str, payload: dict, max_tokens: int) -> dict:
        body = {
            "model": self.config["model"],
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "temperature": float(self.config.get("temperature", 0)),
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": bool(self.config.get("enable_thinking", False))},
        }
        url = self.config["base_url"].rstrip("/") + "/chat/completions"
        timeout = aiohttp.ClientTimeout(total=float(self.config.get("timeout_seconds", 180)))
        attempts = int(self.config.get("max_retries", 1)) + 1
        async with self.semaphore:
            for attempt in range(attempts):
                try:
                    async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as session:
                        async with session.post(url, json=body) as response:
                            response.raise_for_status()
                            result = await response.json()
                    choice = result["choices"][0]
                    if choice.get("finish_reason") != "stop":
                        raise ValueError("Verifier did not finish normally; refusing to convert failure into a reward")
                    text = choice["message"].get("content")
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError("Verifier returned no answer content")
                    return {
                        "text": text,
                        "model": result.get("model"),
                        "usage": result.get("usage", {}),
                        "finish_reason": choice["finish_reason"],
                    }
                except aiohttp.ClientResponseError as error:
                    if error.status < 500 and error.status != 429:
                        raise
                    if attempt + 1 == attempts:
                        raise
                except (aiohttp.ClientError, TimeoutError):
                    if attempt + 1 == attempts:
                        raise
                await asyncio.sleep(min(2**attempt, 4))
        raise RuntimeError("Verifier request exhausted")

    async def solve(self, question: str, evidence: str) -> dict:
        # The interface deliberately has no ground-truth, image, or original-CoT argument.
        return await self._chat(
            PV_PROMPT,
            {"question": question, "observations": evidence},
            int(self.config.get("pv_max_tokens", 4096)),
        )

    async def score(self, question: str, ground_truth: str, prediction: str) -> dict:
        result = await self._chat(
            SVV_PROMPT,
            {"question": question, "ground_truth": ground_truth, "prediction": prediction},
            int(self.config.get("svv_max_tokens", 4096)),
        )
        result["score"] = parse_svv_score(result["text"])
        result["single_box_format"] = len(re.findall(r"\\boxed\s*\{", result["text"])) == 1
        return result
