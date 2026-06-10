"""
LLM wrapper — provider-swappable via LLM_PROVIDER env variable.

Supported providers:
  azure      — Azure OpenAI  (default; preferred for the Microsoft hackathon)
  openai     — Standard OpenAI API  (needs OPENAI_API_KEY)
  compatible — Any OpenAI-compatible endpoint: Groq, Ollama, Together AI, etc.
               (needs OPENAI_COMPATIBLE_BASE_URL + OPENAI_COMPATIBLE_API_KEY)

All calls use JSON mode. Responses are cached to .llm_cache/ during development.
"""

import json
import hashlib
from pathlib import Path
from openai import AzureOpenAI, OpenAI
import app.config as config


_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client

    provider = config.LLM_PROVIDER.lower()

    if provider == "azure":
        _client = AzureOpenAI(
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_key=config.AZURE_OPENAI_API_KEY,
            api_version=config.AZURE_OPENAI_API_VERSION,
        )
    elif provider == "openai":
        _client = OpenAI(api_key=config.OPENAI_API_KEY)
    elif provider == "compatible":
        # Groq, Ollama, Together AI, OpenRouter, etc.
        _client = OpenAI(
            base_url=config.OPENAI_COMPATIBLE_BASE_URL,
            api_key=config.OPENAI_COMPATIBLE_API_KEY or "ollama",  # Ollama ignores the key
        )
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. "
            "Set it to 'azure', 'openai', or 'compatible' in your .env file."
        )
    return _client


def _model_name() -> str:
    provider = config.LLM_PROVIDER.lower()
    if provider == "azure":
        return config.AZURE_OPENAI_DEPLOYMENT
    if provider == "openai":
        return config.OPENAI_MODEL
    return config.OPENAI_COMPATIBLE_MODEL


def _supports_json_mode() -> bool:
    """Returns True if the current provider+model support response_format=json_object."""
    provider = config.LLM_PROVIDER.lower()
    if provider in ("azure", "openai"):
        return True
    # For compatible providers, Groq supports it; plain Ollama does not
    base_url = config.OPENAI_COMPATIBLE_BASE_URL.lower()
    if "groq.com" in base_url:
        return True
    if "openrouter.ai" in base_url:
        return True
    if "together" in base_url:
        return True
    # Ollama and unknown endpoints: rely on prompt instruction instead
    return False


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


def call_llm(system_prompt: str, user_prompt: str, max_retries: int = 3) -> dict:
    """Call the configured LLM with JSON output. Retries on rate limits. Caches to disk."""
    import time

    key = _cache_key(system_prompt, user_prompt)
    cached = _read_cache(key)
    if cached is not None:
        return cached

    client = _get_client()

    # Groq (and some other providers) require the word "json" to appear in the
    # messages when response_format=json_object is set — inject it if missing.
    effective_system = system_prompt
    if _supports_json_mode() and "json" not in (system_prompt + user_prompt).lower():
        effective_system += "\n\nRespond with valid JSON only."

    kwargs = dict(
        model=_model_name(),
        messages=[
            {"role": "system", "content": effective_system},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )
    if _supports_json_mode():
        kwargs["response_format"] = {"type": "json_object"}
    else:
        kwargs["messages"][0]["content"] += "\n\nIMPORTANT: Your entire response must be valid JSON only."

    last_exc = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(**kwargs)
            break
        except Exception as exc:
            last_exc = exc
            err = str(exc).lower()
            # Rate limit → exponential backoff; anything else → raise immediately
            if "rate" in err or "429" in err or "too many" in err:
                wait = 2 ** attempt          # 1s, 2s, 4s
                time.sleep(wait)
                continue
            raise
    else:
        raise last_exc

    content = response.choices[0].message.content.strip()

    # Strip markdown code fences if the model added them
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]

    result = json.loads(content)
    _write_cache(key, result)
    return result
