# Insight Excavator

> Most data tools guess. This one proves.

An agentic analytics pipeline that turns any messy dataset into **statistically verified** insights. Upload a CSV → the system profiles it, cleans it, discovers patterns, and **proves every finding with a real statistical test** before showing it to you. No hallucinated analytics.

Ask it a question in plain English — *"Does customer age affect churn?"* — and it runs the right statistical test and returns a proven answer with p-value, effect size, and chart.

- **Live demo:** *add your Streamlit Cloud URL here*
- **Demo video:** *add link here*

---

## The problem it solves

Most "AI for data" tools pipe a spreadsheet into an LLM and print a confident summary — with no proof it's true. Hallucinated analytics are worse than none.

Insight Excavator closes that trust gap: the LLM is the **creative explorer**, math is the **gatekeeper**. Nothing reaches the user unless a real statistical test confirms it.

---

## What it does

### Automatic discovery pipeline
1. **Profiles** any uploaded dataset (schema, types, nulls, distributions) — pure code, no LLM
2. **Cleans** it — LLM proposes operations from a fixed vocabulary; a deterministic executor applies them; never runs LLM-written code
3. **Discovers** insights in a verify loop — LLM generates structured hypotheses → each is tested by a vetted statistical function → significance gate keeps only real findings
4. **Explains** every surviving finding in plain English, with a chart and the actual numbers (p-value, effect size, n)

### Natural language query engine
Type any question about your data — *"Is there a correlation between tenure and spend?"* — and the engine:
- Parses the question into a structured statistical hypothesis
- Runs the correct test (Pearson, Mann-Whitney, chi-square, etc.)
- Returns a **proven** answer — or honestly reports no significant relationship

This is the "chat with your data" feature most tools hallucinate. This one verifies.

---

## Architecture

```
Upload (CSV / Excel)
        │
        ▼
[1] Profiler          (deterministic, no LLM) → schema, types, nulls, distributions
        │
        ▼
[2] Cleaning agent    (LLM proposes ops → deterministic executor → diff)
        │
        ▼
[3] Insight engine · verify loop
        ┌──────────────────────────────────────────────┐
        │  Hypothesis gen (LLM) → explores the data    │
        │  Verifier → runs a stat test in code         │
        │  Significance gate → keep only if real       │
        │       (discard + retry feeds back)           │
        └──────────────────────────────────────────────┘
        │
        ▼
[4] Narrator + charts (LLM phrases verified numbers only; Plotly charts)
        │
        ▼
Verified insight cards + downloadable report

                     ↕  also available on demand:

[5] NL Query engine  question → parse → stat test → proven answer
```

### Anti-hallucination contract

- The hypothesis LLM **proposes only** — it never reports a result
- The narrator LLM receives **only verified numeric facts** and is forbidden from adding new claims
- Every insight card shows the real p-value, effect size, and n — judges can verify on the spot
- The NL query engine returns **"not significant"** honestly instead of inventing an answer

### Statistical test library

| Hypothesis type | Applies to | Test | Effect size |
|---|---|---|---|
| Correlation | numeric × numeric | Pearson / Spearman | r |
| Group difference | numeric by categorical | t-test / Mann-Whitney / ANOVA / Kruskal-Wallis | Cohen's d / eta² |
| Association | categorical × categorical | chi-square | Cramér's V |
| Trend | numeric over time | linear regression | r (trend strength) |
| Anomaly | numeric | IQR fence / Isolation Forest | anomaly share |
| Missingness | any | correlation of missingness indicator | r |

A finding survives only if `p < 0.05`, effect size clears a per-test minimum, and `n ≥ 30`. Survivors are ranked by surprise score = effect size × confidence. Trivial findings (self-correlations, near-duplicate columns) are dropped.

---

## Tech stack

| Layer | Choice |
|---|---|
| Language | Python 3.11 |
| API | FastAPI + Uvicorn |
| Data & stats | pandas, numpy, scipy, scikit-learn |
| Schemas | Pydantic v2 (structured LLM I/O) |
| LLM | Provider-swappable via `app/llm.py` — Azure OpenAI, OpenAI, Groq, Ollama, OpenRouter |
| Charts & UI | Plotly, Streamlit |
| Tests | pytest (30 tests — library + pipeline) |

---

## LLM provider setup

The app works with any of these — set `LLM_PROVIDER` in your `.env`:

| Provider | `LLM_PROVIDER` | Notes |
|---|---|---|
| OpenAI | `openai` | `gpt-4o-mini` is cheapest |
| Azure OpenAI | `azure` | Best for Microsoft ecosystem |
| Groq | `compatible` | Free tier, very fast |
| Ollama | `compatible` | Fully local, no internet |
| OpenRouter | `compatible` | 100+ models, has free tier |

---

## Setup

**Prerequisites:** Python 3.11+, API key for any supported LLM provider.

```bash
# 1. Clone
git clone https://github.com/debashish-datascience1/insight-excavator.git
cd insight-excavator

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure
cp .env.example .env
# Edit .env — pick a provider and add your key (see .env.example for all options)

# 5. Run the UI
streamlit run ui/streamlit_app.py

# 6. (Optional) Run the API separately
uvicorn app.main:app --reload
```

```bash
# Run the full test suite (30 tests, no LLM required)
pytest
```

---

## Repository structure

```
insight-excavator/
├── app/
│   ├── main.py              # FastAPI — /health, /analyze
│   ├── config.py            # Settings (env + Streamlit secrets)
│   ├── models.py            # Pydantic schemas — Profile, Hypothesis, Finding, RunState
│   ├── llm.py               # Provider-swappable LLM wrapper with disk cache
│   ├── pipeline/
│   │   ├── ingest.py        # Deterministic profiler
│   │   ├── clean.py         # Cleaning agent (fixed-vocab ops only)
│   │   ├── hypothesize.py   # LLM → list[Hypothesis]
│   │   ├── verify.py        # Routes hypotheses to stat functions
│   │   ├── gate.py          # Significance gate + surprise ranking + dedup
│   │   ├── narrate.py       # Phrases verified numbers + builds Plotly charts
│   │   ├── query.py         # NL question → stat verification → proven answer
│   │   └── controller.py   # Orchestrates the full pipeline
│   └── analyses/
│       └── library.py       # Tested statistical functions (the trust layer)
├── ui/
│   └── streamlit_app.py     # Upload → pipeline → insight cards → NL query → report
├── tests/
│   ├── test_library.py      # 22 unit tests for every stat function
│   └── test_pipeline.py     # 8 integration tests (profiler, verifier, gate)
├── data/sample/
│   ├── customer_churn.csv   # Demo dataset (1000 rows, hidden non-obvious patterns)
│   └── generate_sample.py   # Script to regenerate the demo data
├── .env.example             # All provider options documented
└── requirements.txt
```

---

## The builder

Built by **Debashish Mohapatra** — architecture, AI/orchestration design, statistical library, backend, frontend, and deployment. Claude Code used as AI pair-programmer throughout.

---

## License

MIT
