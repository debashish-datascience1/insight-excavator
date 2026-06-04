"""
Azure OpenAI wrapper. Provider-swappable — swap the client init in _get_client().
All calls use JSON mode and are cached to disk during development.
"""

import json
import hashlib
from pathlib import Path
from openai import AzureOpenAI
import app.config as config


_client: AzureOpenAI | None = None


def _get_client() -> AzureOpenAI:
    global _client
    if _client is None:
        _client = AzureOpenAI(
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_key=config.AZURE_OPENAI_API_KEY,
            api_version=config.AZURE_OPENAI_API_VERSION,
        )
    return _client


def _cache_key(system_prompt: str, user_prompt: str) -> str:
    return hashlib.md5(f"{system_prompt}|||{user_prompt}".encode()).hexdigest()


def _read_cache(key: str) -> dict | None:
    if not config.LLM_CACHE_ENABLED:
        return None
    path = Path(config.LLM_CACHE_DIR) / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def _write_cache(key: str, result: dict) -> None:
    if not config.LLM_CACHE_ENABLED:
        return
    Path(config.LLM_CACHE_DIR).mkdir(exist_ok=True)
    (Path(config.LLM_CACHE_DIR) / f"{key}.json").write_text(json.dumps(result))


def call_llm(system_prompt: str, user_prompt: str) -> dict:
    """Call Azure OpenAI with JSON mode. Caches responses to disk."""
    key = _cache_key(system_prompt, user_prompt)
    cached = _read_cache(key)
    if cached is not None:
        return cached

    client = _get_client()
    response = client.chat.completions.create(
        model=config.AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    result = json.loads(response.choices[0].message.content)
    _write_cache(key, result)
    return result
