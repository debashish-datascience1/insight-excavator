# Insight Excavator

> Most data tools guess. This one proves.

An agentic analytics pipeline that turns messy data into **verified** insights. An LLM *proposes* hypotheses about your data; a battery of statistical tests *confirms* them in code. You only ever see findings that are statistically real — each shown with a chart and the actual numbers (effect size, p-value, sample size). No hallucinated analytics.

Built for the **Microsoft Build AI hackathon (HackerEarth)** — theme: *AI Meets Data: From Noise to Insight*.

- **Live demo:** _<add your Streamlit Cloud URL>_
- **Demo video (3 min):** _<add YouTube unlisted link>_

---

## Why it exists

Teams sit on data they can't mine, and the current wave of "AI for data" tools just pipes a spreadsheet into an LLM and prints a confident summary — with no proof it's true. Hallucinated analytics are worse than none. Insight Excavator closes that trust gap: the LLM is the creative explorer, but **math is the gatekeeper**. Nothing reaches the user unless a real statistical test confirms it.

---

## What it does

1. **Profiles** any uploaded dataset (schema, types, nulls, distributions) — pure code, no LLM.
2. **Cleans** it: an LLM proposes operations from a fixed vocabulary; a deterministic executor applies them and shows a before/after diff.
3. **Discovers** insights in a verify loop: an LLM generates structured hypotheses → each is tested by a vetted statistical function → a significance gate keeps only the real ones.
4. **Explains** every surviving finding in plain English (grounded strictly in the computed numbers) with a chart, and assembles a downloadable report.

---

## Architecture overview

```
Messy data (CSV / logs / docs)
        │
        ▼
[1] Profiler            (deterministic, no LLM)  → schema, types, nulls, distributions
        │
        ▼
[2] Cleaning agent      (LLM proposes ops → deterministic executor → validate)
        │
        ▼
[3] Insight engine · verify loop
        ┌─────────────────────────────────────────────┐
        │  Hypothesis gen (LLM)  → explores the data    │
        │  Verifier              → runs a stat in code  │
        │  Significance gate     → keep only if real    │
        │       (discard + retry feeds back)            │
        └───────────────────────────────────────────────┘
        │
        ▼
[4] Narrative + dashboard  (LLM phrases verified numbers only; Plotly charts)
        │
        ▼
Verified insight cards + dashboard + downloadable report
```

**Key design choices**

- **Verify-in-code gate.** The LLM never asserts a result — it proposes a structured hypothesis; a deterministic analysis function computes the answer; a gate keeps it only if statistically significant. This eliminates hallucinated analytics.
- **Grounded narrator.** The narration step receives only verified numeric facts and is instructed to phrase them, never to add new claims.
- **Deterministic brackets.** Profiling and cleaning execution are pure code, so the pipeline can't crash on a bad generation.

### Statistical test library

| Hypothesis type | Applies to | Test | Effect size |
|---|---|---|---|
| Correlation | numeric × numeric | Pearson / Spearman | r |
| Group difference | numeric by categorical | t-test / ANOVA (non-parametric fallbacks) | Cohen's d / eta² |
| Association | categorical × categorical | chi-square | Cramér's V |
| Trend | numeric over time | linear regression / Mann-Kendall | slope |
| Anomaly | numeric | IQR / z-score / Isolation Forest | count + share |

A finding survives only if `p < 0.05`, the effect size clears a per-test minimum, and the sample is adequate. Survivors are ranked by a surprise score and trivial findings (e.g. duplicate-column correlations) are dropped.

---

## Tech stack & AI tools used

- **LLM:** Azure OpenAI (provider-swappable via `app/llm.py`)
- **Backend / orchestration:** Python 3.11, FastAPI
- **Data & stats:** pandas, numpy, scipy, scikit-learn
- **Schemas:** Pydantic v2 (structured LLM I/O)
- **Charts & UI:** Plotly, Streamlit (deployed on Streamlit Community Cloud)
- **Testing:** pytest
- **Built with:** Claude Code (AI pair-programmer) for implementation across all phases

---

## Setup instructions

**Prerequisites:** Python 3.11+, an Azure OpenAI resource with a deployed model (or any OpenAI-compatible provider).

```bash
# 1. Clone
git clone https://github.com/<your-username>/insight-excavator.git
cd insight-excavator

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# then edit .env and add:
#   AZURE_OPENAI_ENDPOINT=...
#   AZURE_OPENAI_API_KEY=...
#   AZURE_OPENAI_DEPLOYMENT=...

# 5. Run the API
uvicorn app.main:app --reload

# 6. In a second terminal, run the UI
streamlit run ui/streamlit_app.py
```

Then open the Streamlit URL, upload a dataset, and watch the pipeline run.

```bash
# Run tests (the statistical library is fully unit-tested)
pytest
```

---

## Dependencies

Pinned in `requirements.txt`. Core: `fastapi`, `uvicorn`, `streamlit`, `pandas`, `numpy`, `scipy`, `scikit-learn`, `plotly`, `pydantic`, `openai` (Azure-compatible client), `python-dotenv`, `pytest`.

---

## Repository structure

```
insight-excavator/
├── app/
│   ├── main.py          # FastAPI app + routes
│   ├── llm.py           # Azure OpenAI wrapper (swappable)
│   ├── models.py        # Pydantic schemas
│   ├── pipeline/        # ingest, clean, hypothesize, verify, gate, narrate, controller
│   └── analyses/        # the tested statistical library
├── ui/streamlit_app.py  # upload → insights UI
├── tests/               # unit + end-to-end tests
├── data/sample/         # demo datasets
└── requirements.txt
```

---

## The builder

Built solo by **_<your name>_** — sole developer covering architecture, AI/orchestration design, backend, frontend, and deployment, with Claude Code as an AI pair-programmer.

---

## License

MIT (or your choice).
