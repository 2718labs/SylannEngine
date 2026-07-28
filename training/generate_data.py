"""SylannEngine 嵌入式感知模型 — 数据生成 Pipeline.

用 Haiku 批量生成训练数据：
1. 合成对话文本（中英文混合）
2. 对每条文本生成情感标注向量

输出格式: JSONL, 每行 {"text": "...", "lang": "zh|en", "emotion": {...}, "meta": {...}}

Usage:
    # Provide provider API keys via environment variables (never hardcode them):
    export SYLANNE_GPT55_API_KEY=...   # provider api.aylovelle.top
    export SYLANNE_MIMO_API_KEY=...    # provider token-plan-cn.xiaomimimo.com
    python generate_data.py --n 10000 --output data/train.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYNTH_SYSTEM_PROMPT = """You are a dialogue data generator for an affective computing system.
Generate realistic single-turn messages that a user might send to an AI companion.
The messages should cover diverse emotional scenarios.

Output EXACTLY one JSON object per request with these fields:
{
  "text": "the message text",
  "lang": "zh" or "en",
  "scenario": "brief scenario description",
  "emotion_raw": {
    "valence": float [-1, 1],  // negative to positive
    "arousal": float [0, 1],   // calm to excited
    "dominance": float [0, 1], // submissive to dominant
    "warmth": float [0, 1],    // cold to warm
    "vulnerability": float [0, 1],  // guarded to vulnerable
    "hostility": float [0, 1],  // friendly to hostile
    "engagement": float [0, 1],  // disengaged to engaged
    "surprise": float [0, 1]   // expected to surprising
  },
  "intent": "one of: express_joy, seek_comfort, vent_anger, share_sadness, show_gratitude, set_boundary, seek_connection, express_love, test_trust, withdraw, greet, farewell, small_talk, deep_talk, conflict, reconcile"
}

Be diverse. Cover: casual chat, emotional disclosure, conflict, intimacy, boundaries, silence-breaking, etc.
Vary sentence length (1 word to 3 sentences). Include slang, typos, emoji occasionally."""

SYNTH_USER_PROMPTS = {
    "zh_casual": "Generate a casual Chinese message. Scenario: everyday small talk.",
    "zh_emotional": "Generate an emotionally charged Chinese message. Scenario: user expressing strong feelings.",
    "zh_conflict": "Generate a Chinese message showing conflict or frustration with the AI.",
    "zh_intimate": "Generate an intimate/warm Chinese message showing closeness.",
    "zh_boundary": "Generate a Chinese message where user sets boundaries or rejects interaction.",
    "zh_vulnerable": "Generate a vulnerable Chinese message where user opens up about pain.",
    "en_casual": "Generate a casual English message. Scenario: everyday small talk.",
    "en_emotional": "Generate an emotionally charged English message. Scenario: user expressing strong feelings.",
    "en_conflict": "Generate an English message showing conflict or frustration with the AI.",
    "en_intimate": "Generate an intimate/warm English message showing closeness.",
    "en_boundary": "Generate an English message where user sets boundaries or rejects interaction.",
    "en_vulnerable": "Generate a vulnerable English message where user opens up about pain.",
}

# Scenario weights (more casual, less extreme)
SCENARIO_WEIGHTS = {
    "zh_casual": 20,
    "zh_emotional": 15,
    "zh_conflict": 10,
    "zh_intimate": 12,
    "zh_boundary": 8,
    "zh_vulnerable": 10,
    "en_casual": 10,
    "en_emotional": 8,
    "en_conflict": 5,
    "en_intimate": 6,
    "en_boundary": 4,
    "en_vulnerable": 5,
}


def weighted_scenario_choice() -> str:
    scenarios = list(SCENARIO_WEIGHTS.keys())
    weights = list(SCENARIO_WEIGHTS.values())
    return random.choices(scenarios, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# API client (multi-provider)
# ---------------------------------------------------------------------------

# Provider configs. API keys are read from environment variables at runtime —
# NEVER hardcode credentials here (they end up in git history). Each provider
# names the env var that holds its key; unset providers are skipped.
PROVIDERS = [
    {
        "name": "gpt5.5",
        "base_url": "https://api.aylovelle.top/v1",
        "api_key_env": "SYLANNE_GPT55_API_KEY",
        "model": "gpt-5.5",
        "concurrency": 8,
        "temperature": 0.9,
        "top_p": 0.95,
    },
    {
        "name": "mimo-v2.5",
        "base_url": "https://token-plan-cn.xiaomimimo.com/v1",
        "api_key_env": "SYLANNE_MIMO_API_KEY",
        "model": "MiMo-V2.5",
        "concurrency": 8,
        "temperature": 0.85,
        "top_p": 0.9,
    },
    {
        "name": "mimo-v2.5-pro",
        "base_url": "https://token-plan-cn.xiaomimimo.com/v1",
        "api_key_env": "SYLANNE_MIMO_API_KEY",
        "model": "MiMo-V2.5-Pro",
        "concurrency": 7,
        "temperature": 0.8,
        "top_p": 0.85,
    },
]


async def call_llm(
    client,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.9,
    top_p: float = 0.95,
    max_retries: int = 3,
) -> dict | None:
    """Call LLM via OpenAI-compatible API and parse JSON response."""
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                max_tokens=500,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                top_p=top_p,
            )
            text = response.choices[0].message.content.strip()
            # Extract JSON from response
            if text.startswith("{"):
                return json.loads(text)
            # Try to find JSON in response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except json.JSONDecodeError:
            continue
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2**attempt)
            else:
                print(f"  [{model}] Failed: {e}")
    return None


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


async def generate_batch(
    clients: list[dict],
    batch_size: int = 50,
) -> list[dict]:
    """Generate a batch of training samples using multiple providers concurrently."""
    # Distribute batch across providers proportionally to their concurrency
    total_concurrency = sum(c["concurrency"] for c in clients)
    tasks = []

    for provider in clients:
        share = max(1, int(batch_size * provider["concurrency"] / total_concurrency))
        semaphore = provider["_semaphore"]

        async def gen_one(p=provider, sem=semaphore) -> dict | None:
            async with sem:
                scenario = weighted_scenario_choice()
                user_prompt = SYNTH_USER_PROMPTS[scenario]
                result = await call_llm(
                    p["_client"],
                    p["model"],
                    SYNTH_SYSTEM_PROMPT,
                    user_prompt,
                    temperature=p["temperature"],
                    top_p=p["top_p"],
                )
                if result and "text" in result and "emotion_raw" in result:
                    result["_scenario_key"] = scenario
                    result["_provider"] = p["name"]
                    return result
                return None

        tasks.extend([gen_one() for _ in range(share)])

    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]


async def main():
    parser = argparse.ArgumentParser(
        description="Generate training data for embedded perception model"
    )
    parser.add_argument("--n", type=int, default=10000, help="Number of samples to generate")
    parser.add_argument("--output", type=str, default="data/train.jsonl", help="Output file")
    parser.add_argument("--batch-size", type=int, default=60, help="Batch size per round")
    args = parser.parse_args()

    from openai import AsyncOpenAI

    # Initialize all providers whose API key is present in the environment.
    clients = []
    for p in PROVIDERS:
        api_key = os.environ.get(p["api_key_env"], "")
        if not api_key:
            print(f"Skipping provider {p['name']}: set {p['api_key_env']} to enable it")
            continue
        client_obj = AsyncOpenAI(api_key=api_key, base_url=p["base_url"])
        clients.append(
            {
                **p,
                "_client": client_obj,
                "_semaphore": asyncio.Semaphore(p["concurrency"]),
            }
        )
    if not clients:
        required = ", ".join(sorted({p["api_key_env"] for p in PROVIDERS}))
        raise SystemExit(f"No provider API keys found. Set at least one of: {required}")
    print(f"Providers: {', '.join(p['name'] for p in clients)}")
    print(f"Total concurrency: {sum(p['concurrency'] for p in clients)}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_generated = 0
    start_time = time.time()

    # Resume from existing file
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            total_generated = sum(1 for _ in f)
        print(f"Resuming from {total_generated} existing samples")

    with open(output_path, "a", encoding="utf-8") as f:
        while total_generated < args.n:
            remaining = args.n - total_generated
            batch_size = min(args.batch_size, remaining)

            batch = await generate_batch(clients, batch_size)

            for sample in batch:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            f.flush()

            total_generated += len(batch)
            elapsed = time.time() - start_time
            rate = total_generated / elapsed if elapsed > 0 else 0
            eta = (args.n - total_generated) / rate if rate > 0 else 0

            print(
                f"  [{total_generated}/{args.n}] "
                f"batch={len(batch)} "
                f"rate={rate:.1f}/s "
                f"ETA={eta:.0f}s"
            )

    print(f"\nDone! Generated {total_generated} samples in {time.time() - start_time:.0f}s")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
