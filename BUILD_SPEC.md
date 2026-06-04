# Insight Excavator — Build Spec

> **How to use this file:** Drop it at the root of an empty repo and tell Claude Code:
> *"Read BUILD_SPEC.md and build this project. Work phase by phase. After each phase, run the acceptance check and tell me the result before starting the next phase."*
> Implement phases in order. Do not skip ahead. Keep the demo path reliable over clever.

---

## 0. What this is for

This is a submission for the **Microsoft Build AI hackathon (hosted on HackerEarth)**.

- **Prize pool:** ₹6,00,000.
- **Mandatory deliverables:** a working live link (up ≥30 days), a 3-minute live-demo video, a public GitHub repo with README (≤3 pages), and a 10-slide PDF deck.
- **Judging weights (total 100):**
  - AI Integration & Intelligence Design — **25%**
  - System Architecture & Engineering Quality — **25%**
  - Communication, Presentation & UX — **15%**
  - Prototype Readiness & Scalability — **15%**
  - Problem Depth & Product Clarity — **10%**
  - Market Understanding & Product Fit — **10%**

**Design implication:** 50% of the score is "is the AI design sophisticated AND is the system well-engineered." 30% is "does it visibly work and look clean." Everything below is optimized for that. The product MUST run end-to-end on camera and produce a *surprising, verified* result.

---

## 1. The themes (full context)

The hackathon has six themes. We target **AI Meets Data**, but the engine can be re-pitched toward others if needed (see §1.7).

### 1.1 AI at Work: Productivity & Teamwork Reimagined
Smart assistants, intelligent workflows, real-time knowledge sharing, conversation summaries, tools that ensure no idea or task is lost — for individuals and distributed teams.

### 1.2 Security in the Agentic Future
Monitoring frameworks, defense mechanisms, and trust architectures protecting agentic systems from prompt injection, identity spoofing, unauthorized access, and adversarial misuse.

### 1.3 Agentic Web
Autonomous agents that browse, navigate sites, extract info, complete multi-step transactions, and recover from failure without hand-holding.

### 1.4 AI Meets Data: From Noise to Insight  ← **TARGET THEME**
Take raw, messy, unstructured data and turn it into something teams can act on: intelligent cleaning, automated enrichment, pattern discovery, analytics pipelines that surface the signal in the noise. Success line from the brief: *"I had no idea that was in our data."*

### 1.5 Agent Swarms
Orchestrating multiple agents (planners, retrievers, executors, validators) that self-organize to solve multi-step problems no single agent could handle. Distributed, containerized, scalable.

### 1.6 AI-Powered Production Function: Reinventing Work
AI-native CI/CD, intelligent quality gates, automated testing/deployment, AI-augmented project management, adaptive process orchestration — building the production function ground-up for AI.

### 1.7 How our project maps / can pivot
- **Primary (AI Meets Data):** the whole product is built for this. The verify-loop directly produces the "I had no idea" moment.
- **Secondary framing if asked to pivot:** the multi-stage verify loop is itself a small **Agent Swarm** (hypothesis / verifier / gate / narrator agents) — lean into that language in the deck for bonus "intelligence design" credit. Pointed at team logs or chat exports it serves **AI at Work**. The "verify in code, never trust the LLM's claim" principle is also a **Security/trust** story (defends against hallucinated analytics).

---

## 2. Product concept

**Name (working):** Insight Excavator — rename freely.
**One-liner:** *Most data tools guess. This one proves.*

Drop in a messy dataset → the system profiles it, cleans it, then runs an **agentic discovery loop**: an LLM *proposes* hypotheses about what might be interesting, but **every insight only survives if it is confirmed by real statistical computation run in code.** The LLM is the creative explorer; math is the gatekeeper. Output is a set of plain-English insight cards, each backed by a chart and the actual numbers (effect size, p-value, sample size), plus a downloadable report.

**Why this wins:** most "AI + data" hackathon entries pipe a CSV into an LLM and print a summary — which hallucinates. Our differentiator is that **no insight is shown unless it is statistically verified.** That single choice is the story for both the 25% AI-Integration bucket and the 25% Engineering bucket, and it is a memorable pitch line.

---

## 3. Architecture

Pipeline of specialized stages, orchestrated by an explicit controller (state object passed through). Deterministic stages bracket the LLM stages so the clever part never fights bad input.

```
Messy data in (CSV / logs / docs)
        │
        ▼
[1] Profiler            (deterministic, NO LLM)  → schema, types, nulls, distributions
        │
        ▼
[2] Cleaning agent      (LLM proposes ops → deterministic executor → validate)  → before/after diff
        │
        ▼
[3] INSIGHT ENGINE  · verify loop  (the core, the differentiator)
        ┌─────────────────────────────────────────────┐
        │  Hypothesis gen (LLM)  → explores the data    │
        │           │                                   │
        │           ▼                                   │
        │  Verifier  → runs a stat from a fixed library │
        │           │     in a sandbox (scipy/pandas)   │
        │           ▼                                   │
        │  Significance gate → keep only if real        │
        │           │  (discard + retry feeds back)     │
        └───────────┼───────────────────────────────────┘
                    ▼
[4] Narrative + dashboard  (LLM phrases VERIFIED numbers only; Plotly charts)
        │
        ▼
Verified insight cards + live dashboard + downloadable report
```

### Key engineering decisions (call these out in the deck)
1. **Verify-in-code gate:** the LLM never asserts a finding. It proposes a *structured hypothesis*; a deterministic analysis function computes the result; a gate keeps it only if statistically significant. This eliminates hallucinated analytics.
2. **Narrator is grounded:** the narrative LLM receives ONLY the verified numeric facts and is instructed to phrase them, never to introduce new claims.
3. **Deterministic brackets:** profiling and cleaning execution are pure code, not LLM, so the demo can't crash on a bad generation.

---

## 4. Tech stack

| Layer | Choice | Notes |
|---|---|---|
| Language | Python 3.11 | |
| API | FastAPI + Uvicorn | endpoints for upload / profile / clean / analyze / results |
| Data | pandas, numpy | core dataframe ops |
| Stats | scipy, scikit-learn | tests + IsolationForest for anomalies |
| Schemas | Pydantic v2 | structured LLM I/O (Profile, Hypothesis, Finding, Operation) |
| LLM | **Azure OpenAI** | Microsoft event → stack fit scores on Market Fit. Provider swappable via `llm.py`. Use JSON-mode / structured output. |
| Charts | Plotly | per-finding charts |
| UI | **Streamlit** | fastest path to a working demoable UI; deploys free to Streamlit Community Cloud, stays up past the 30-day rule |
| Tests | pytest | the analysis library MUST be unit-tested (it's the trust story) |
| Config | python-dotenv | keys in `.env`, never committed |

**Do NOT** spend time on a custom React frontend. UX is 15% and Streamlit with polished insight cards is enough. The 50% lives in the engine.

---

## 5. Orchestration approach

Use an **explicit controller** (`pipeline/controller.py`) holding a single `RunState` object that flows through stages. This is reliable and easy to demo.

- Optionally mention LangGraph in the deck as the "could-scale-to" path, but **ship the explicit controller** — fewer moving parts on demo day.
- Each stage is a pure-ish function: `stage(state) -> state`. Log timing per stage (good architecture evidence, and gives a nice progress UI).

---

## 6. The analysis library (`analyses/library.py`) — build this FIRST and TEST it

This is the heart of the trust story and is fully testable without any LLM. Each function takes a dataframe + column args and returns a structured result `{test, stat, effect_size, p_value, n, summary_numbers}`.

| Hypothesis type | Applies to | Test | Effect size |
|---|---|---|---|
| `correlation` | numeric × numeric | Pearson (or Spearman if non-normal) | r |
| `group_difference` | numeric by categorical (2 groups) | independent t-test (Mann-Whitney if non-normal) | Cohen's d |
| `group_difference` | numeric by categorical (>2 groups) | one-way ANOVA (Kruskal-Wallis fallback) | eta-squared |
| `association` | categorical × categorical | chi-square test of independence | Cramér's V |
| `trend` | numeric over datetime | linear regression / Mann-Kendall | slope + significance |
| `anomaly` | numeric (uni/multi) | IQR / z-score / IsolationForest | count + share |
| `missingness` | any | missingness pattern + correlation | quality flags |

**Significance gate rules:** keep a finding only if `p_value < 0.05` AND `|effect_size| >= type-specific minimum` AND `n` is adequate. Rank survivors by a **surprise score** = effect size × confidence, and **drop trivial findings** (e.g. near-perfect correlation between duplicate/derived columns, or a column correlated with itself).

---

## 7. Repository structure

```
insight-excavator/
├── README.md                  # written in Phase 9 (≤3 pages: desc, setup, deps, architecture, AI tools, team)
├── BUILD_SPEC.md              # this file
├── requirements.txt
├── .env.example               # AZURE_OPENAI_* keys, no real secrets
├── .gitignore                 # .env, __pycache__, data/uploads
├── app/
│   ├── main.py                # FastAPI app + routes
│   ├── config.py              # settings, thresholds, model name
│   ├── models.py              # Pydantic: Profile, Operation, Hypothesis, Finding, RunState
│   ├── llm.py                 # Azure OpenAI wrapper; JSON-structured calls; provider-swappable
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── ingest.py          # load file + deterministic profiler
│   │   ├── clean.py           # cleaning agent (LLM plan) + deterministic executor + diff
│   │   ├── hypothesize.py     # LLM → list[Hypothesis]
│   │   ├── verify.py          # runs hypotheses against analyses.library
│   │   ├── gate.py            # significance gate + surprise ranking + dedupe
│   │   ├── narrate.py         # verified findings → insight cards + Plotly charts
│   │   └── controller.py      # orchestrates the full RunState pipeline + loop
│   └── analyses/
│       ├── __init__.py
│       └── library.py         # the fixed, tested stat functions (§6)
├── ui/
│   └── streamlit_app.py       # upload → progress → insight cards + charts → download report
├── tests/
│   ├── test_library.py        # unit tests for every stat function (synthetic data, known answers)
│   └── test_pipeline.py       # end-to-end on a small fixture CSV
└── data/
    └── sample/                # demo datasets (see §10)
```

---

## 8. Build roadmap (phase by phase, with acceptance checks)

Implement in this exact order. Each phase ends with a check Claude Code must confirm before continuing.

**Phase 0 — Scaffold.** Create repo structure, `requirements.txt`, `.env.example`, `.gitignore`, a FastAPI app with a `/health` route.
*Done when:* `uvicorn app.main:app` runs and `/health` returns `{"status":"ok"}`.

**Phase 1 — Ingest + Profiler (deterministic, no LLM).** Load CSV/Excel into pandas; infer schema, dtypes, null counts, basic distributions, candidate categorical vs numeric vs datetime columns. Return a `Profile` model.
*Done when:* uploading a sample CSV returns a complete profile JSON; no LLM involved.

**Phase 2 — Analysis library + tests (no LLM).** Implement all functions in §6 in `analyses/library.py`. Unit-test each on synthetic data with known statistical answers.
*Done when:* `pytest tests/test_library.py` passes for every test type.

**Phase 3 — Cleaning agent.** LLM proposes a list of `Operation`s drawn from a FIXED vocabulary (dedupe, coerce type, trim/normalize strings, parse dates, handle nulls). A deterministic executor applies each op to pandas. Produce a before/after diff (row counts, changed cells).
*Done when:* a deliberately messy CSV is cleaned and the diff is shown; executor never runs LLM-written raw code.

**Phase 4 — Hypothesis generator.** LLM receives the profile + a data sample and returns `list[Hypothesis]` as strict JSON (validated by Pydantic). Each hypothesis names a type from §6 and the columns involved.
*Done when:* generator returns ≥10 valid, parseable hypotheses for the sample dataset.

**Phase 5 — Verifier + Gate + Loop controller.** For each hypothesis, route to the matching library function, run it, collect results. Apply the significance gate, drop trivial/duplicate findings, rank by surprise. Loop a configurable N rounds (round 2 can target leads from round 1).
*Done when:* the pipeline outputs a ranked list of `Finding`s, each carrying test name, effect size, p-value, n.

**Phase 6 — Narrate + charts.** Pass ONLY verified numbers to the LLM; it phrases each finding as a one-sentence insight (no new claims allowed). Build a Plotly chart per finding. Assemble a dashboard payload + a downloadable HTML/markdown report.
*Done when:* each insight card shows a plain-English statement, a chart, and the underlying stats.

**Phase 7 — Streamlit UI.** Upload widget → live progress through stages → insight cards with charts → "download report" button. Polish the cards specifically (they're what judges see). Light theming, sentence case, clean spacing.
*Done when:* a non-technical person can upload a file and read insights without touching the API.

**Phase 8 — Deploy + demo dataset.** Deploy the Streamlit app to Streamlit Community Cloud (public URL, stays up ≥30 days). Lock in a demo dataset that yields a genuinely counterintuitive finding (§10).
*Done when:* the public URL works from a fresh browser and surfaces the "whoa" insight reliably.

**Phase 9 — Deliverables.** Write README (≤3 pages: description, setup, dependencies, architecture overview, AI tools used, team + roles). Draft the 10-slide deck (problem, solution, architecture diagram, AI integration = the verify loop, demo screenshots, team). Record the 3-minute live-walkthrough video.
*Done when:* all four mandatory deliverables exist and meet format limits.

---

## 9. Anti-hallucination contract (enforce throughout)

- The hypothesis LLM may PROPOSE only; it never reports results.
- The narrator LLM receives verified numeric facts and is instructed: *phrase these numbers in plain English; do not add, infer, or estimate anything not present in the inputs.*
- Every insight card displays its real p-value, effect size, and n. If a judge asks "is this made up?", the numbers are right there.
- This contract IS the engineering-quality and AI-design narrative — make it explicit in code comments and the deck.

---

## 10. Demo dataset (do this early — it makes or breaks the video)

Pick a public dataset with a **non-obvious hidden relationship** so the on-camera reveal lands. Good hunting grounds: Kaggle, data.gov, UCI ML repo. Look for a dataset where a surprising correlation or group difference exists (e.g., a counterintuitive driver of churn, an unexpected segment difference, a hidden seasonal trend). Verify the surprising finding manually FIRST, then make sure the pipeline rediscovers it. A working pipeline on a boring dataset wins nothing; a working pipeline that says something surprising wins the room.

---

## 11. De-risking notes for Claude Code

- **Ship the fixed-library version (Phases 2–6 as written).** Do NOT let the LLM write and execute arbitrary Python in the demo build — it crashes unpredictably. The LLM selects and parameterizes vetted functions only.
- Optional stretch (only if Phases 0–9 are done with time to spare): a sandboxed free-form code-execution verifier. Keep it behind a feature flag; never on the demo path.
- Build the skeleton end-to-end on ONE hardcoded dataset by the end of Phase 1–2 so there's always something runnable.
- Cache LLM responses during development to save cost and speed up iteration.
- Keep all stage timings logged — they double as the progress UI and as architecture evidence.

---

## 12. Acceptance summary (definition of done for the whole project)

- [ ] Public Streamlit URL, works from a clean browser, stays up ≥30 days.
- [ ] Upload → profile → clean → verified insights → downloadable report, end to end.
- [ ] Every insight is backed by a real statistical test (visible p-value, effect size, n).
- [ ] At least one genuinely surprising verified finding on the demo dataset.
- [ ] Public GitHub repo with ≤3-page README (desc, setup, deps, architecture, AI tools, team).
- [ ] 10-slide PDF deck including the architecture diagram and the verify-loop AI-integration slide.
- [ ] 3-minute MP4 live walkthrough (≥720p) — real run, not slides.
