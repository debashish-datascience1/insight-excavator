import os
from dotenv import load_dotenv

load_dotenv()


def _get(key: str, default: str = "") -> str:
    """Read from env first, then Streamlit secrets (when running on Streamlit Cloud)."""
    val = os.getenv(key, "")
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:
        return default


# ── Provider selection ─────────────────────────────────────────
# Options: azure | openai | compatible
LLM_PROVIDER = _get("LLM_PROVIDER", "azure")

# ── Azure OpenAI ───────────────────────────────────────────────
AZURE_OPENAI_ENDPOINT   = _get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY    = _get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = _get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
AZURE_OPENAI_API_VERSION = _get("AZURE_OPENAI_API_VERSION", "2024-02-01")

# ── Standard OpenAI ────────────────────────────────────────────
OPENAI_API_KEY = _get("OPENAI_API_KEY")
OPENAI_MODEL   = _get("OPENAI_MODEL", "gpt-4o-mini")

# ── OpenAI-compatible (Groq / Ollama / Together / OpenRouter) ──
OPENAI_COMPATIBLE_BASE_URL = _get("OPENAI_COMPATIBLE_BASE_URL", "http://localhost:11434/v1")
OPENAI_COMPATIBLE_API_KEY  = _get("OPENAI_COMPATIBLE_API_KEY")
OPENAI_COMPATIBLE_MODEL    = _get("OPENAI_COMPATIBLE_MODEL", "llama3")

# ── Analysis thresholds ────────────────────────────────────────
SIGNIFICANCE_THRESHOLD   = float(_get("SIGNIFICANCE_THRESHOLD", "0.05"))
MIN_EFFECT_CORRELATION   = 0.1
MIN_EFFECT_COHENS_D      = 0.2
MIN_EFFECT_CRAMERS_V     = 0.1
MIN_EFFECT_ETA_SQUARED   = 0.01
MIN_EFFECT_ANOMALY_SHARE = 0.02
MIN_N                    = 30

VERIFY_LOOP_ROUNDS        = int(_get("VERIFY_LOOP_ROUNDS", "2"))
MAX_HYPOTHESES_PER_ROUND  = int(_get("MAX_HYPOTHESES_PER_ROUND", "15"))

# ── Cache (disable on Streamlit Cloud to avoid filesystem issues) ──
LLM_CACHE_DIR     = _get("LLM_CACHE_DIR", ".llm_cache")
LLM_CACHE_ENABLED = _get("LLM_CACHE_ENABLED", "true").lower() == "true"
