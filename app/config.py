import os
from dotenv import load_dotenv

load_dotenv()

# ── Provider selection ─────────────────────────────────────────
# Set LLM_PROVIDER to one of: azure | openai | compatible
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "azure")

# ── Azure OpenAI ───────────────────────────────────────────────
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

# ── Standard OpenAI ────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ── OpenAI-compatible (Groq / Ollama / Together / OpenRouter) ──
OPENAI_COMPATIBLE_BASE_URL = os.getenv("OPENAI_COMPATIBLE_BASE_URL", "http://localhost:11434/v1")
OPENAI_COMPATIBLE_API_KEY = os.getenv("OPENAI_COMPATIBLE_API_KEY", "")
OPENAI_COMPATIBLE_MODEL = os.getenv("OPENAI_COMPATIBLE_MODEL", "llama3")

SIGNIFICANCE_THRESHOLD = float(os.getenv("SIGNIFICANCE_THRESHOLD", "0.05"))
MIN_EFFECT_CORRELATION = 0.1
MIN_EFFECT_COHENS_D = 0.2
MIN_EFFECT_CRAMERS_V = 0.1
MIN_EFFECT_ETA_SQUARED = 0.01
MIN_EFFECT_ANOMALY_SHARE = 0.02
MIN_N = 30

VERIFY_LOOP_ROUNDS = int(os.getenv("VERIFY_LOOP_ROUNDS", "2"))
MAX_HYPOTHESES_PER_ROUND = int(os.getenv("MAX_HYPOTHESES_PER_ROUND", "15"))

LLM_CACHE_DIR = os.getenv("LLM_CACHE_DIR", ".llm_cache")
LLM_CACHE_ENABLED = os.getenv("LLM_CACHE_ENABLED", "true").lower() == "true"
